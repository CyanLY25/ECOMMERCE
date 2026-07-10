# Reporte de Validación Estadística

Fecha de generación: 2026-07-09 16:15:06

## Resumen Ejecutivo

- **Métrica evaluada**: rmse
- **Nivel de significancia (α)**: 0.05
- **Modelo con mejor rendimiento**: mlp

---

## 1. Prueba de Friedman

### Resultados
- **Estadístico χ²**: 10.8800
- **Valor p**: 0.027946
- **α**: 0.05

### Conclusión
Se rechaza la hipótesis nula (p = 0.027946 < α = 0.05). Existen diferencias estadísticamente significativas entre al menos dos modelos.

### Ranking Promedio
1. mlp: 1.000
2. gru: 3.000
3. cnn_lstm: 3.400
4. cnn_gru: 3.800
5. lstm: 3.800


---

## 2. Test Post-Hoc de Nemenyi

- **Diferencia Crítica (CD)**: 2.728

### Comparaciones Significativas
- cnn_gru vs mlp
- lstm vs mlp


---

## 3. Test de Wilcoxon con Corrección de Bonferroni

- **Número de comparaciones**: 10

### Resultados
| model_a   | model_b   |   statistic |   p_value |   cohens_d | cohens_interpretation   |   cliffs_delta | cliffs_interpretation   |   mean_a |   mean_b |    mean_diff |   p_value_bonferroni | significant   | interpretation                                                              |
|:----------|:----------|------------:|----------:|-----------:|:------------------------|---------------:|:------------------------|---------:|---------:|-------------:|---------------------:|:--------------|:----------------------------------------------------------------------------|
| cnn_gru   | cnn_lstm  |           7 |    1      |  0.0181637 | Despreciable            |           0.04 | Despreciable            |  97.2424 |  97.2424 |  1.85298e-05 |                1     | False         | No hay diferencias estadísticamente significativas entre cnn_gru y cnn_lstm |
| cnn_gru   | gru       |           3 |    0.3125 |  0.481319  | Pequeño                 |           0.04 | Despreciable            |  97.2424 |  97.2393 |  0.00311397  |                1     | False         | No hay diferencias estadísticamente significativas entre cnn_gru y gru      |
| cnn_gru   | lstm      |           6 |    0.8125 |  0.0102532 | Despreciable            |           0.04 | Despreciable            |  97.2424 |  97.2424 |  5.8326e-06  |                1     | False         | No hay diferencias estadísticamente significativas entre cnn_gru y lstm     |
| cnn_gru   | mlp       |           0 |    0.0625 |  2.10701   | Grande                  |           0.2  | Pequeño                 |  97.2424 |  96.3504 |  0.892043    |                0.625 | False         | No hay diferencias estadísticamente significativas entre cnn_gru y mlp      |
| cnn_lstm  | gru       |           5 |    0.625  |  0.47811   | Pequeño                 |           0.04 | Despreciable            |  97.2424 |  97.2393 |  0.00309544  |                1     | False         | No hay diferencias estadísticamente significativas entre cnn_lstm y gru     |
| cnn_lstm  | lstm      |           6 |    0.8125 | -0.0124372 | Despreciable            |          -0.04 | Despreciable            |  97.2424 |  97.2424 | -1.26972e-05 |                1     | False         | No hay diferencias estadísticamente significativas entre cnn_lstm y lstm    |
| cnn_lstm  | mlp       |           0 |    0.0625 |  2.10433   | Grande                  |           0.2  | Pequeño                 |  97.2424 |  96.3504 |  0.892024    |                0.625 | False         | No hay diferencias estadísticamente significativas entre cnn_lstm y mlp     |
| gru       | lstm      |           4 |    0.4375 | -0.445558  | Pequeño                 |          -0.12 | Despreciable            |  97.2393 |  97.2424 | -0.00310814  |                1     | False         | No hay diferencias estadísticamente significativas entre gru y lstm         |
| gru       | mlp       |           0 |    0.0625 |  2.11566   | Grande                  |           0.2  | Pequeño                 |  97.2393 |  96.3504 |  0.888929    |                0.625 | False         | No hay diferencias estadísticamente significativas entre gru y mlp          |
| lstm      | mlp       |           0 |    0.0625 |  2.1054    | Grande                  |           0.2  | Pequeño                 |  97.2424 |  96.3504 |  0.892037    |                0.625 | False         | No hay diferencias estadísticamente significativas entre lstm y mlp         |

---

## 4. Intervalos de Confianza (95%)

| model    |    mean |     std |   ci_low |   ci_high |   n |
|:---------|--------:|--------:|---------:|----------:|----:|
| mlp      | 96.3504 | 124.565 | -58.3171 |   251.018 |   5 |
| lstm     | 97.2424 | 124.163 | -56.9265 |   251.411 |   5 |
| gru      | 97.2393 | 124.165 | -56.9317 |   251.41  |   5 |
| cnn_lstm | 97.2424 | 124.163 | -56.9261 |   251.411 |   5 |
| cnn_gru  | 97.2424 | 124.163 | -56.9267 |   251.412 |   5 |

---

## 5. Ranking de Modelos

| model    |    rmse |   rank |
|:---------|--------:|-------:|
| mlp      | 96.3504 |      1 |
| gru      | 97.2393 |      2 |
| cnn_lstm | 97.2424 |      3 |
| lstm     | 97.2424 |      4 |
| cnn_gru  | 97.2424 |      5 |

---

## 6. Conclusiones Generales

1. **Mejor modelo**: mlp
2. **Significancia global**: Sí

---

*Report generado automáticamente por el módulo de validación estadística.*
