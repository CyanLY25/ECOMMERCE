import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from ia.config.config import AIConfig
from ia.training.common import (
    set_seed,
    load_datasets,
    separate_features_target,
    create_sequences,
    get_callbacks,
    calculate_metrics,
    save_metrics,
    save_history,
    plot_training_history,
    get_optimizer
)
from ia.utils.logger import setup_logger


@tf.keras.utils.register_keras_serializable(package="tft")
def tft_slice_feature(x, idx):
    """Selecciona la característica `idx` conservando la dimensión temporal."""
    return x[:, :, idx:idx + 1]


@tf.keras.utils.register_keras_serializable(package="tft")
def tft_stack_variables(tensors):
    """Apila las representaciones de las variables en un nuevo eje."""
    return tf.stack(tensors, axis=2)


@tf.keras.utils.register_keras_serializable(package="tft")
def tft_expand_last_dim(x):
    """Añade una dimensión final (para poder multiplicar por difusión)."""
    return tf.expand_dims(x, axis=-1)


@tf.keras.utils.register_keras_serializable(package="tft")
def tft_sum_variables(x):
    """Suma ponderada a lo largo del eje de variables."""
    return tf.reduce_sum(x, axis=2)


@tf.keras.utils.register_keras_serializable(package="tft")
def tft_last_timestep(x):
    """Extrae la representación del último paso de tiempo de la secuencia."""
    return x[:, -1, :]


def get_tft_custom_objects() -> Dict[str, Any]:
    """Objetos necesarios para deserializar un TFT en un proceso nuevo."""
    functions = {
        "tft_slice_feature": tft_slice_feature,
        "tft_stack_variables": tft_stack_variables,
        "tft_expand_last_dim": tft_expand_last_dim,
        "tft_sum_variables": tft_sum_variables,
        "tft_last_timestep": tft_last_timestep,
    }
    # Keras puede guardar las funciones registradas con el prefijo del paquete.
    return {
        **functions,
        **{f"tft>{name}": function for name, function in functions.items()},
    }


def load_tft_model(model_path: Path) -> tf.keras.Model:
    """Carga un modelo TFT incluyendo sus funciones personalizadas."""
    return tf.keras.models.load_model(
        model_path,
        custom_objects=get_tft_custom_objects(),
    )


class TFTModel:
    """
    Clase para implementar y entrenar una versión compacta del
    Temporal Fusion Transformer (TFT) para predicción de demanda.

    Implementa los bloques principales propuestos por Lim et al. (2021)
    adaptados a un escenario de regresión univariante de un solo paso
    (igual que el resto de modelos secuenciales del proyecto):

        1. Variable Selection Network (VSN): aprende, para cada paso de
           tiempo, qué tan relevante es cada característica de entrada
           y combina las características ponderándolas por esa
           relevancia.
        2. LSTM Encoder (procesamiento local): captura patrones
           secuenciales de corto plazo sobre la representación generada
           por la VSN, con una conexión residual con compuerta (GLU).
        3. Bloque de auto-atención multi-cabeza (Interpretable
           Multi-Head Attention): captura dependencias de largo plazo
           entre los distintos pasos de tiempo de la ventana.
        4. Position-wise Feed-Forward (Gated Residual Network): procesa
           la salida de la atención con otra conexión residual con
           compuerta.
        5. Capa de salida: se toma la representación del último paso de
           tiempo y se proyecta a un único valor (regresión).

    Todas las conexiones residuales usan Gated Residual Networks (GRN) y
    Gated Linear Units (GLU), tal como en el paper original del TFT.
    """

    def __init__(self, config: AIConfig):
        """
        Inicializa el modelo TFT.

        Args:
            config: Objeto de configuración.
        """
        self.config = config
        self.logger = setup_logger("tft", config.TFT_LOG_PATH)
        self.model: tf.keras.Model = None

    # ------------------------------------------------------------------
    # Bloques del Temporal Fusion Transformer
    # ------------------------------------------------------------------
    def _glu(self, x: tf.Tensor, units: int, name: str) -> tf.Tensor:
        """
        Gated Linear Unit (GLU): combina una proyección lineal con una
        compuerta sigmoide que controla cuánta información pasa.
        """
        linear = tf.keras.layers.Dense(units, name=f"{name}_glu_linear")(x)
        gate = tf.keras.layers.Dense(units, activation="sigmoid", name=f"{name}_glu_gate")(x)
        return tf.keras.layers.Multiply(name=f"{name}_glu_mul")([linear, gate])

    def _grn(self, x: tf.Tensor, hidden_size: int, dropout_rate: float,
             name: str, output_size: Optional[int] = None) -> tf.Tensor:
        """
        Gated Residual Network (GRN): bloque base del TFT que permite a
        la red decidir cuánta transformación no lineal aplicar, con una
        conexión residual y normalización de capa.
        """
        if output_size is None:
            output_size = x.shape[-1]

        # Proyección del residual si la dimensión de salida no coincide
        if x.shape[-1] != output_size:
            residual = tf.keras.layers.Dense(output_size, name=f"{name}_residual_proj")(x)
        else:
            residual = x

        hidden = tf.keras.layers.Dense(hidden_size, name=f"{name}_dense1")(x)
        hidden = tf.keras.layers.Activation("elu", name=f"{name}_elu")(hidden)
        hidden = tf.keras.layers.Dense(hidden_size, name=f"{name}_dense2")(hidden)
        hidden = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_dropout")(hidden)

        gated = self._glu(hidden, output_size, name=name)
        out = tf.keras.layers.Add(name=f"{name}_add")([residual, gated])
        out = tf.keras.layers.LayerNormalization(name=f"{name}_norm")(out)
        return out

    def _variable_selection_network(self, inputs: tf.Tensor, num_features: int,
                                     hidden_size: int, dropout_rate: float) -> tf.Tensor:
        """
        Variable Selection Network (VSN): proyecta cada característica a un
        espacio oculto común, calcula pesos de selección (softmax) por paso
        de tiempo y combina las características transformadas mediante esos
        pesos. Esto permite que el modelo enfatice, en cada instante, las
        variables más relevantes para la predicción.
        """
        transformed = []
        for i in range(num_features):
            # Selecciona la i-ésima característica manteniendo la dimensión
            feature_i = tf.keras.layers.Lambda(
                tft_slice_feature,
                arguments={"idx": i},
                name=f"vsn_slice_{i}"
            )(inputs)
            # Proyección lineal a hidden_size (aplicada igual en cada paso de tiempo)
            proj_i = tf.keras.layers.Dense(hidden_size, name=f"vsn_proj_{i}")(feature_i)
            # Enriquecimiento no lineal propio de la variable
            grn_i = self._grn(proj_i, hidden_size, dropout_rate, name=f"vsn_grn_{i}")
            transformed.append(grn_i)

        # Apilar en un nuevo eje de "variables": (batch, window, num_features, hidden_size)
        stacked = tf.keras.layers.Lambda(
            tft_stack_variables, name="vsn_stack"
        )(transformed)

        # Concatenar todas las proyecciones para calcular los pesos de selección
        flat_concat = tf.keras.layers.Concatenate(axis=-1, name="vsn_concat")(transformed)
        selection_logits = self._grn(
            flat_concat, hidden_size, dropout_rate,
            name="vsn_selection", output_size=num_features
        )
        selection_weights = tf.keras.layers.Softmax(axis=-1, name="vsn_softmax")(selection_logits)
        # (batch, window, num_features) -> (batch, window, num_features, 1)
        selection_weights = tf.keras.layers.Lambda(
            tft_expand_last_dim, name="vsn_weights_expand"
        )(selection_weights)

        # Combinar: suma ponderada de las variables transformadas
        weighted = tf.keras.layers.Multiply(name="vsn_weighted")([stacked, selection_weights])
        combined = tf.keras.layers.Lambda(
            tft_sum_variables, name="vsn_sum"
        )(weighted)

        return combined

    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Construye la arquitectura del modelo TFT.

        Args:
            input_shape: Forma de los datos de entrada (window_size, num_features).

        Returns:
            Modelo Keras compilado.
        """
        # Limpiar sesión Keras
        tf.keras.backend.clear_session()

        if len(input_shape) != 2 or any(dim is None or int(dim) <= 0 for dim in input_shape):
            raise ValueError(
                "TFT requiere input_shape=(ventana, características) con dimensiones positivas"
            )

        window_size, num_features = input_shape
        hidden_size = self.config.TFT_HIDDEN_SIZE
        num_heads = self.config.TFT_NUM_HEADS
        dropout_rate = self.config.TFT_DROPOUT

        if hidden_size <= 0 or num_heads <= 0:
            raise ValueError("TFT_HIDDEN_SIZE y TFT_NUM_HEADS deben ser mayores que cero")
        if not 0 <= dropout_rate < 1:
            raise ValueError("TFT_DROPOUT debe estar en el intervalo [0, 1)")

        inputs = tf.keras.Input(shape=input_shape)

        # 1) Variable Selection Network
        selected = self._variable_selection_network(
            inputs, num_features, hidden_size, dropout_rate
        )

        # 2) Procesamiento local (LSTM) con conexión residual con compuerta
        lstm_out = tf.keras.layers.LSTM(
            hidden_size, return_sequences=True, name="tft_lstm_encoder"
        )(selected)
        gated_lstm = self._glu(lstm_out, hidden_size, name="tft_lstm_gate")
        temporal_features = tf.keras.layers.Add(name="tft_lstm_add")([selected, gated_lstm])
        temporal_features = tf.keras.layers.LayerNormalization(
            name="tft_lstm_norm"
        )(temporal_features)

        # 3) Auto-atención multi-cabeza (dependencias de largo plazo)
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(hidden_size // num_heads, 1),
            dropout=dropout_rate,
            name="tft_attention"
        )(temporal_features, temporal_features)
        gated_attn = self._glu(attn_out, hidden_size, name="tft_attn_gate")
        attn_features = tf.keras.layers.Add(name="tft_attn_add")([temporal_features, gated_attn])
        attn_features = tf.keras.layers.LayerNormalization(name="tft_attn_norm")(attn_features)

        # 4) Position-wise Feed-Forward (GRN) con conexión residual con compuerta
        ff = self._grn(attn_features, hidden_size, dropout_rate, name="tft_position_ff")
        gated_ff = self._glu(ff, hidden_size, name="tft_ff_gate")
        decoder_out = tf.keras.layers.Add(name="tft_ff_add")([attn_features, gated_ff])
        decoder_out = tf.keras.layers.LayerNormalization(name="tft_ff_norm")(decoder_out)

        # 5) Tomar el último paso de tiempo y proyectar a la salida
        last_step = tf.keras.layers.Lambda(
            tft_last_timestep, name="tft_last_step"
        )(decoder_out)
        last_step = tf.keras.layers.Dropout(dropout_rate, name="tft_output_dropout")(last_step)
        outputs = tf.keras.layers.Dense(1, activation="linear", name="tft_output")(last_step)

        # Construir y compilar el modelo
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="TFT")
        optimizer = get_optimizer(self.config)
        optimizer.learning_rate = self.config.TFT_LEARNING_RATE
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"], run_eagerly=False)

        self.model = model
        self.logger.info("Arquitectura TFT construida exitosamente")
        model.summary(print_fn=self.logger.info)

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float], float]:
        """
        Entrena el modelo TFT.

        Args:
            X_train: Datos de entrenamiento (secuencias).
            y_train: Etiquetas de entrenamiento.
            X_val: Datos de validación (secuencias).
            y_val: Etiquetas de validación.

        Returns:
            Tupla con (historial_entrenamiento, métricas, tiempo_entrenamiento).
        """
        set_seed(self.config)
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO ENTRENAMIENTO TFT")
        self.logger.info("=" * 60)

        self._validate_training_data(X_train, y_train, "entrenamiento")
        self._validate_training_data(X_val, y_val, "validación")
        if X_train.shape[1:] != X_val.shape[1:]:
            raise ValueError(
                "Entrenamiento y validación deben tener la misma ventana y número de características"
            )

        # Construir modelo
        self.build_model((X_train.shape[1], X_train.shape[2]))

        # Obtener callbacks
        callbacks = get_callbacks(self.config, "tft")

        # Entrenar
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.TFT_EPOCHS,
            batch_size=self.config.TFT_BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        end_time = time.time()
        training_time = end_time - start_time

        # Cargar el mejor modelo
        self.model = load_tft_model(self.config.TFT_MODEL_PATH)

        # Evaluar
        self.logger.info("Evaluando modelo TFT...")
        y_pred = self.model.predict(X_val, verbose=0).flatten()
        metrics = calculate_metrics(y_val, y_pred)
        metrics['loss'] = float(self.model.evaluate(X_val, y_val, verbose=0)[0])
        metrics['val_loss'] = float(min(history.history['val_loss']))
        metrics['training_time'] = float(training_time)
        metrics['epochs_run'] = int(len(history.epoch))

        # Guardar
        save_history(history, self.config.TFT_HISTORY_PATH)
        save_metrics(metrics, self.config.TFT_METRICS_PATH)
        plot_training_history(history.history, self.config.FIGURES_DIR, "tft")

        self.logger.info(f"TFT entrenado en {training_time:.2f} segundos")
        self.logger.info(f"Métricas: {metrics}")
        self.logger.info("=" * 60)

        return history.history, metrics, training_time

    @staticmethod
    def _validate_training_data(X: np.ndarray, y: np.ndarray, split_name: str) -> None:
        """Valida dimensiones y valores antes de iniciar un entrenamiento costoso."""
        if X.ndim != 3:
            raise ValueError(
                f"Los datos de {split_name} de TFT deben ser 3D; se recibió {X.shape}"
            )
        if y.ndim != 1:
            raise ValueError(
                f"El target de {split_name} de TFT debe ser 1D; se recibió {y.shape}"
            )
        if len(X) == 0 or len(y) == 0:
            raise ValueError(f"No hay secuencias disponibles para {split_name}")
        if len(X) != len(y):
            raise ValueError(
                f"Cantidad diferente de entradas ({len(X)}) y targets ({len(y)}) en {split_name}"
            )
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"Se encontraron NaN o infinitos en {split_name}")
