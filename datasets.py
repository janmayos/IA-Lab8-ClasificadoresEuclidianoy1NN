import re
import math
import random

def crear_matriz_linea_linea_archivo(archivo):
    matriz_file = []
    # Abre el archivo en modo lectura
    with open(archivo, 'r') as file:
        # Lee cada linea a linea del archivo
        for line in file:
            matriz_file.append(line.strip())
    return matriz_file

def imprimir_linea_matriz(matriz_file):
    for line in matriz_file:
        print(line)

def procesar_data(matriz_file):
    matriz_process = []
    for line in matriz_file:
        if line.find(",") != -1:
            datosline = line.split(",")
            datoslinecast = []
            for dato in datosline:
                if re.match("[0-9]+.[0-9]*",dato):
                    datoslinecast.append(float(dato))
                else:
                    datoslinecast.append(dato)
            matriz_process.append((datoslinecast))
    return matriz_process

def separar_caracteristicas_clases(matri_process):
    caracteristicas = [row[:-1] for row in matri_process]  
    clases = [row[-1] for row in matri_process]   
    return caracteristicas,clases

#Funcion de distancia euclidiana el cual compara dos listas de información y con 
# ayuda de zip permite crear un interador de tuplas el cual combina cada dato con 
# su respectivo par ordenado de ciertos valores
#La función zip() toma dos o más iterables (en este caso, point1 y point2) y los combina en un iterador de tuplas. Cada tupla contiene elementos de las posiciones correspondientes de los iterables.
# Ejemplo: Si point1 = [1, 2, 3] y point2 = [4, 5, 6], zip(point1, point2) producirá un iterador que genera las tuplas (1, 4), (2, 5), (3, 6).

def euclidean_distancia(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))


def clasificador_euclidiano(caracteristicas, clases, punto_prueba):
    """Clasifica un punto de prueba utilizando la distancia euclidiana."""
    
    # Inicializar la distancia mínima como infinito
    min_distance = float('inf')
    clase_predicha = None
    
    # Recorrer cada punto de entrenamiento
    for i in range(len(caracteristicas)):
        distancia = euclidean_distancia(caracteristicas[i], punto_prueba)
        
        # Si la distancia actual es menor que la mínima, actualizar
        if distancia < min_distance:
            min_distance = distancia
            clase_predicha = clases[i]  # Asignar la clase correspondiente
    
    return clase_predicha

def clasificador_1nn(caracteristicas, clases, punto_prueba):
    """Clasifica un punto de prueba utilizando el clasificador 1-NN."""
    
    # Inicializar la distancia mínima como infinito
    min_distance = float('inf')
    clase_determinada = None
    
    # Recorrer cada punto de entrenamiento
    for i in range(len(caracteristicas)):
        distancia = euclidean_distancia(caracteristicas[i], punto_prueba)
        
        # Si la distancia actual es menor que la mínima, actualizar
        if distancia < min_distance:
            min_distance = distancia
            clase_determinada = clases[i]
    
    return clase_determinada

# Función para calcular la matriz de confusión
def matriz_confusion(true_labels, pred_labels):
    cm = {}
    # Inicializa las claves de las clases si no existen
    for true, pred in zip(true_labels, pred_labels):
        if true not in cm:
            cm[true] = {}
        if pred not in cm[true]:
            cm[true][pred] = 0
        cm[true][pred] += 1
    return cm

# Función para calcular el Accuracy
def accuracy(y_true, y_pred):
    """Calcula el accuracy (precisión) del clasificador."""
    correct = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])
    return correct / len(y_true)

# Función para dividir el conjunto de datos en Hold Out 70/30
def hold_out_70_30(caracteristicas, clases, clasificador, test_size=0.3):
    """Validación Hold Out 70/30"""
    datacombinada = list(zip(caracteristicas, clases))
    random.shuffle(datacombinada)
    split_cantidad_datos = int(len(datacombinada) * (1 - test_size))
    entrenamiento_data = datacombinada[:split_cantidad_datos]
    test_data = datacombinada[split_cantidad_datos:]
    
    entrenamiento_features, entrenamiento_labels = zip(*entrenamiento_data)
    test_features, test_labels = zip(*test_data)
    
    # Predecir
    y_pred = [clasificador(entrenamiento_features, entrenamiento_labels, x) for x in test_features]
    
    # Calcular desempeño
    acc = accuracy(test_labels, y_pred)
    cm = matriz_confusion(test_labels, y_pred)
    return acc, cm

import random

def k_fold_cross_validation(caracteristicas, clases, clasificador, k=10):
    """Validación 10-Fold Cross-Validation"""
    # Combina las características y clases en una lista de tuplas
    datacombinada = list(zip(caracteristicas, clases))
    
    # Mezcla aleatoriamente los datos
    random.shuffle(datacombinada)
    
    # Calcula el tamaño de cada pliegue (fold)
    fold_size = len(datacombinada) // k
    
    accuracies = []  # Para almacenar las precisiones de cada pliegue
    confusion_matrices = []  # Para almacenar las matrices de confusión de cada pliegue
    
    for i in range(k):
        # Divide en k pliegues: test y train
        test_data = datacombinada[i * fold_size : (i + 1) * fold_size]
        train_data = datacombinada[:i * fold_size] + datacombinada[(i + 1) * fold_size:]
        
        # Separa las características y las etiquetas para entrenamiento y prueba
        entrenamiento_features, entrenamiento_labels = zip(*train_data)
        test_features, test_labels = zip(*test_data)
        
        # Predecir
        y_pred = [clasificador(entrenamiento_features, entrenamiento_labels, x) for x in test_features]
        
        # Calcular desempeño
        acc = accuracy(test_labels, y_pred)
        cm = matriz_confusion(test_labels, y_pred)
        
        # Almacena el desempeño de este pliegue
        accuracies.append(acc)
        confusion_matrices.append(cm)
    
    # Promedio de las precisiones
    avg_acc = sum(accuracies) / len(accuracies)
    
    # Promedio de las matrices de confusión
    avg_cm = {}
    for cm in confusion_matrices:
        for true_class in cm:
            for pred_class in cm[true_class]:
                if (true_class, pred_class) not in avg_cm:
                    avg_cm[(true_class, pred_class)] = 0
                avg_cm[(true_class, pred_class)] += cm[true_class][pred_class]
    
    return avg_acc, avg_cm

import random

# Función para Leave-One-Out
def leave_one_out(caracteristicas, clases, clasificador):
    """Validación Leave-One-Out"""
    # Combina las características y clases en una lista de tuplas
    datacombinada = list(zip(caracteristicas, clases))
    
    accuracies = []  # Para almacenar las precisiones de cada iteración
    confusion_matrices = []  # Para almacenar las matrices de confusión de cada iteración
    
    for i in range(len(datacombinada)):
        # Divide en datos de prueba (un solo dato) y datos de entrenamiento (el resto)
        test_data = [datacombinada[i]]
        train_data = datacombinada[:i] + datacombinada[i+1:]
        
        # Separa las características y las etiquetas para entrenamiento y prueba
        entrenamiento_features, entrenamiento_labels = zip(*train_data)
        test_features, test_labels = zip(*test_data)
        
        # Predecir
        y_pred = [clasificador(entrenamiento_features, entrenamiento_labels, x) for x in test_features]
        
        # Calcular desempeño
        acc = accuracy(test_labels, y_pred)
        cm = matriz_confusion(test_labels, y_pred)
        
        # Almacena el desempeño de esta iteración
        accuracies.append(acc)
        confusion_matrices.append(cm)
    
    # Promedio de las precisiones
    avg_acc = sum(accuracies) / len(accuracies)
    
    # Promedio de las matrices de confusión
    avg_cm = {}
    for cm in confusion_matrices:
        for true_class in cm:
            for pred_class in cm[true_class]:
                if (true_class, pred_class) not in avg_cm:
                    avg_cm[(true_class, pred_class)] = 0
                avg_cm[(true_class, pred_class)] += cm[true_class][pred_class]
    
    return avg_acc, avg_cm


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
