# Reporte de Validación Estadística

Fecha de generación: 2026-07-16 14:38:28

## Resumen Ejecutivo

- **Métrica evaluada**: rmse
- **Nivel de significancia (α)**: 0.05
- **Modelo con mejor rendimiento**: mlp

---

## 1. Prueba de Friedman

### Resultados
- **Estadístico χ²**: 12.3200
- **Valor p**: 0.015124
- **α**: 0.05

### Conclusión
Se rechaza la hipótesis nula (p = 0.015124 < α = 0.05). Existen diferencias estadísticamente significativas entre al menos dos modelos.

### Ranking Promedio
1. mlp: 1.000
2. lstm: 2.600
3. cnn_lstm: 3.600
4. cnn_gru: 3.800
5. gru: 4.000


---

## 2. Test Post-Hoc de Nemenyi

- **Diferencia Crítica (CD)**: 2.728

### Comparaciones Significativas
- cnn_gru vs mlp
- gru vs mlp


---

## 3. Test de Wilcoxon con Corrección de Bonferroni

- **Número de comparaciones**: 10

### Resultados
| model_a   | model_b   |   statistic |   p_value |   cohens_d | cohens_interpretation   |   cliffs_delta | cliffs_interpretation   |   mean_a |   mean_b |   mean_diff |   p_value_bonferroni | significant   | interpretation                                                              |
|:----------|:----------|------------:|----------:|-----------:|:------------------------|---------------:|:------------------------|---------:|---------:|------------:|---------------------:|:--------------|:----------------------------------------------------------------------------|
| cnn_gru   | cnn_lstm  |           3 |    0.3125 |   0.621857 | Mediano                 |           0.04 | Despreciable            |  71.8927 |  71.6612 |   0.231484  |                1     | False         | No hay diferencias estadísticamente significativas entre cnn_gru y cnn_lstm |
| cnn_gru   | gru       |           4 |    0.4375 |   0.373918 | Pequeño                 |           0.04 | Despreciable            |  71.8927 |  71.817  |   0.0757572 |                1     | False         | No hay diferencias estadísticamente significativas entre cnn_gru y gru      |
| cnn_gru   | lstm      |           3 |    0.3125 |   0.918387 | Grande                  |           0.04 | Despreciable            |  71.8927 |  71.5803 |   0.312369  |                1     | False         | No hay diferencias estadísticamente significativas entre cnn_gru y lstm     |
| cnn_gru   | mlp       |           0 |    0.0625 |   3.18761  | Grande                  |           0.2  | Pequeño                 |  71.8927 |  59.4074 |  12.4853    |                0.625 | False         | No hay diferencias estadísticamente significativas entre cnn_gru y mlp      |
| cnn_lstm  | gru       |           5 |    0.625  |  -0.456125 | Pequeño                 |          -0.12 | Despreciable            |  71.6612 |  71.817  |  -0.155727  |                1     | False         | No hay diferencias estadísticamente significativas entre cnn_lstm y gru     |
| cnn_lstm  | lstm      |           0 |    0.0625 |   0.854911 | Grande                  |           0.2  | Pequeño                 |  71.6612 |  71.5803 |   0.0808844 |                0.625 | False         | No hay diferencias estadísticamente significativas entre cnn_lstm y lstm    |
| cnn_lstm  | mlp       |           0 |    0.0625 |   3.14727  | Grande                  |           0.2  | Pequeño                 |  71.6612 |  59.4074 |  12.2538    |                0.625 | False         | No hay diferencias estadísticamente significativas entre cnn_lstm y mlp     |
| gru       | lstm      |           2 |    0.1875 |   0.745408 | Mediano                 |           0.12 | Despreciable            |  71.817  |  71.5803 |   0.236612  |                1     | False         | No hay diferencias estadísticamente significativas entre gru y lstm         |
| gru       | mlp       |           0 |    0.0625 |   3.30008  | Grande                  |           0.2  | Pequeño                 |  71.817  |  59.4074 |  12.4095    |                0.625 | False         | No hay diferencias estadísticamente significativas entre gru y mlp          |
| lstm      | mlp       |           0 |    0.0625 |   3.15653  | Grande                  |           0.2  | Pequeño                 |  71.5803 |  59.4074 |  12.1729    |                0.625 | False         | No hay diferencias estadísticamente significativas entre lstm y mlp         |

---

## 4. Intervalos de Confianza (95%)

| model    |    mean |     std |   ci_low |   ci_high |   n |
|:---------|--------:|--------:|---------:|----------:|----:|
| mlp      | 59.4074 | 5.98885 |  51.9713 |   66.8436 |   5 |
| lstm     | 71.5803 | 9.48108 |  59.808  |   83.3527 |   5 |
| gru      | 71.817  | 9.3322  |  60.2295 |   83.4044 |   5 |
| cnn_lstm | 71.6612 | 9.5159  |  59.8457 |   83.4768 |   5 |
| cnn_gru  | 71.8927 | 9.4992  |  60.0979 |   83.6875 |   5 |

---

## 5. Ranking de Modelos

| model    |    rmse |   rank |
|:---------|--------:|-------:|
| mlp      | 59.4074 |      1 |
| lstm     | 71.5803 |      2 |
| cnn_lstm | 71.6612 |      3 |
| gru      | 71.817  |      4 |
| cnn_gru  | 71.8927 |      5 |

---

## 6. Conclusiones Generales

1. **Mejor modelo**: mlp
2. **Significancia global**: Sí

---

*Report generado automáticamente por el módulo de validación estadística.*
