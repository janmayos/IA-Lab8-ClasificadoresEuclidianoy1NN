from Clasificador import leave_one_out,k_fold_cross_validation,hold_out_70_30,clasificador_1nn,clasificador_euclidiano
from Datasets import crear_matriz_linea_linea_archivo,procesar_data,separar_caracteristicas_clases,imprimir_linea_matriz


if __name__ == "__main__":
    matriz_file = crear_matriz_linea_linea_archivo('iris/bezdekIris.data')
    #imprimir_linea_matriz(matriz_file)
    matriz_process = procesar_data(matriz_file)
    #imprimir_linea_matriz(matriz_process)
    caracteristicas,clases = separar_caracteristicas_clases(matriz_process)
    imprimir_linea_matriz(caracteristicas)
    
    punto_prueba = [6.0, 3.1, 4.9, 2.3]
    clase = clasificador_euclidiano(caracteristicas, clases, punto_prueba)
    print(f'La clase predicha es: {clase}')
    clase = clasificador_1nn(caracteristicas, clases, punto_prueba)
    print(f'La clase predicha es: {clase}')


    # Validación Hold Out 70/30
    acc_ho, cm_ho = hold_out_70_30(caracteristicas, clases, clasificador_1nn)
    print(f"Hold Out 70/30 - Accuracy: {acc_ho:.2f}")
    print(f"Matriz de Confusión: {cm_ho}")

    # Validación 10-Fold Cross-Validation
    acc_cv, cm_cv = k_fold_cross_validation(caracteristicas, clases, clasificador_1nn, k=10)
    print(f"10-Fold Cross-Validation - Accuracy: {acc_cv:.2f}")
    print(f"Matriz de Confusión: {cm_cv}")

    acc_lo, cm_lo = leave_one_out(caracteristicas, clases, clasificador_1nn)
    print(f"Leave-One-Out - Accuracy: {acc_lo:.2f}")
    print(f"Matriz de Confusión: {cm_lo}")
