# Reporte de Validación Estadística

Fecha de generación: 2026-07-19 11:58:14

## Resumen Ejecutivo

- **Métrica evaluada**: rmse
- **Nivel de significancia (α)**: 0.05
- **Modelo con mejor rendimiento**: mlp

---

## 1. Prueba de Friedman

### Resultados
- **Estadístico χ²**: 12.6571
- **Valor p**: 0.026813
- **α**: 0.05

### Conclusión
Se rechaza la hipótesis nula (p = 0.026813 < α = 0.05). Existen diferencias estadísticamente significativas entre al menos dos modelos.

### Ranking Promedio
1. mlp: 1.000
2. lstm: 3.000
3. cnn_lstm: 4.000
4. cnn_gru: 4.200
5. gru: 4.400
6. tft: 4.400


---

## 2. Test Post-Hoc de Nemenyi

- **Diferencia Crítica (CD)**: 3.372

### Comparaciones Significativas
- gru vs mlp
- mlp vs tft


---

## 3. Test de Wilcoxon con Corrección de Bonferroni

- **Número de comparaciones**: 15

### Resultados
| model_a   | model_b   |   statistic |   p_value |   cohens_d | cohens_interpretation   |   cliffs_delta | cliffs_interpretation   |   mean_a |   mean_b |   mean_diff |   p_value_bonferroni | significant   | interpretation                                                              |
|:----------|:----------|------------:|----------:|-----------:|:------------------------|---------------:|:------------------------|---------:|---------:|------------:|---------------------:|:--------------|:----------------------------------------------------------------------------|
| cnn_gru   | cnn_lstm  |           3 |    0.3125 |   0.621857 | Mediano                 |           0.04 | Despreciable            |  71.8927 |  71.6612 |   0.231484  |               1      | False         | No hay diferencias estadísticamente significativas entre cnn_gru y cnn_lstm |
| cnn_gru   | gru       |           4 |    0.4375 |   0.373918 | Pequeño                 |           0.04 | Despreciable            |  71.8927 |  71.817  |   0.0757572 |               1      | False         | No hay diferencias estadísticamente significativas entre cnn_gru y gru      |
| cnn_gru   | lstm      |           3 |    0.3125 |   0.918387 | Grande                  |           0.04 | Despreciable            |  71.8927 |  71.5803 |   0.312369  |               1      | False         | No hay diferencias estadísticamente significativas entre cnn_gru y lstm     |
| cnn_gru   | mlp       |           0 |    0.0625 |   3.18761  | Grande                  |           0.2  | Pequeño                 |  71.8927 |  59.4074 |  12.4853    |               0.9375 | False         | No hay diferencias estadísticamente significativas entre cnn_gru y mlp      |
| cnn_gru   | tft       |           5 |    0.625  |  -0.275571 | Pequeño                 |          -0.04 | Despreciable            |  71.8927 |  74.9228 |  -3.03006   |               1      | False         | No hay diferencias estadísticamente significativas entre cnn_gru y tft      |
| cnn_lstm  | gru       |           5 |    0.625  |  -0.456125 | Pequeño                 |          -0.12 | Despreciable            |  71.6612 |  71.817  |  -0.155727  |               1      | False         | No hay diferencias estadísticamente significativas entre cnn_lstm y gru     |
| cnn_lstm  | lstm      |           0 |    0.0625 |   0.854911 | Grande                  |           0.2  | Pequeño                 |  71.6612 |  71.5803 |   0.0808844 |               0.9375 | False         | No hay diferencias estadísticamente significativas entre cnn_lstm y lstm    |
| cnn_lstm  | mlp       |           0 |    0.0625 |   3.14727  | Grande                  |           0.2  | Pequeño                 |  71.6612 |  59.4074 |  12.2538    |               0.9375 | False         | No hay diferencias estadísticamente significativas entre cnn_lstm y mlp     |
| cnn_lstm  | tft       |           5 |    0.625  |  -0.293394 | Pequeño                 |          -0.04 | Despreciable            |  71.6612 |  74.9228 |  -3.26155   |               1      | False         | No hay diferencias estadísticamente significativas entre cnn_lstm y tft     |
| gru       | lstm      |           2 |    0.1875 |   0.745408 | Mediano                 |           0.12 | Despreciable            |  71.817  |  71.5803 |   0.236612  |               1      | False         | No hay diferencias estadísticamente significativas entre gru y lstm         |
| gru       | mlp       |           0 |    0.0625 |   3.30008  | Grande                  |           0.2  | Pequeño                 |  71.817  |  59.4074 |  12.4095    |               0.9375 | False         | No hay diferencias estadísticamente significativas entre gru y mlp          |
| gru       | tft       |           5 |    0.625  |  -0.284134 | Pequeño                 |          -0.04 | Despreciable            |  71.817  |  74.9228 |  -3.10582   |               1      | False         | No hay diferencias estadísticamente significativas entre gru y tft          |
| lstm      | mlp       |           0 |    0.0625 |   3.15653  | Grande                  |           0.2  | Pequeño                 |  71.5803 |  59.4074 |  12.1729    |               0.9375 | False         | No hay diferencias estadísticamente significativas entre lstm y mlp         |
| lstm      | tft       |           5 |    0.625  |  -0.300192 | Pequeño                 |          -0.04 | Despreciable            |  71.5803 |  74.9228 |  -3.34243   |               1      | False         | No hay diferencias estadísticamente significativas entre lstm y tft         |
| mlp       | tft       |           0 |    0.0625 |  -1.57393  | Grande                  |          -0.2  | Pequeño                 |  59.4074 |  74.9228 | -15.5153    |               0.9375 | False         | No hay diferencias estadísticamente significativas entre mlp y tft          |

---

## 4. Intervalos de Confianza (95%)

| model    |    mean |      std |   ci_low |   ci_high |   n |
|:---------|--------:|---------:|---------:|----------:|----:|
| mlp      | 59.4074 |  5.98885 |  51.9713 |   66.8436 |   5 |
| lstm     | 71.5803 |  9.48108 |  59.808  |   83.3527 |   5 |
| gru      | 71.817  |  9.3322  |  60.2295 |   83.4044 |   5 |
| cnn_lstm | 71.6612 |  9.5159  |  59.8457 |   83.4768 |   5 |
| cnn_gru  | 71.8927 |  9.4992  |  60.0979 |   83.6875 |   5 |
| tft      | 74.9228 | 11.6086  |  60.5088 |   89.3368 |   5 |

---

## 5. Ranking de Modelos

| model    |    rmse |   rank |
|:---------|--------:|-------:|
| mlp      | 59.4074 |      1 |
| lstm     | 71.5803 |      2 |
| cnn_lstm | 71.6612 |      3 |
| gru      | 71.817  |      4 |
| cnn_gru  | 71.8927 |      5 |
| tft      | 74.9228 |      6 |

---

## 6. Conclusiones Generales

1. **Mejor modelo**: mlp
2. **Significancia global**: Sí

---

*Report generado automáticamente por el módulo de validación estadística.*
