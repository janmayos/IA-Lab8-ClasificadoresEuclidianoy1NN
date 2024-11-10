
def crear_matriz_linea_linea_archivo(archivo):
    matriz_file = []
    # Abre el archivo en modo lectura
    with open(archivo, 'r') as file:
        # Lee cada linea a linea del archivo
        for line in file:
            matriz_file.append(line.strip())
    return matriz_file

def imprimir_linea_matriz(matriz):
    for line in matriz_file:
        print(line)

if __name__ == "__main__":
    matriz_file = crear_matriz_linea_linea_archivo('iris/bezdekIris.data')
    imprimir_linea_matriz(matriz_file)
    
