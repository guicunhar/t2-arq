/*
GPU
Versão Convencional:
- Transforma a matriz A e o vetor B de forma convencional

Versão 1: 
- Baseado na ideia original com tentativa de sincronização

Versão 2:
- Sem necessidade de sincronização porque utiliza um vetor auxiliar para armazenar os dados compartilhados entre as threads

Versão 3:
- Usa uma matriz auxiliar para armazenar o resultado

Versão 4:
- Distribuiu uniformemente os elementos da matriz pelos processadores
*/

#ifndef COMUM
#define COMUM

/// macro de acesso à matriz
#define matriz(mA,linha,coluna,tamanho) (mA[(linha)*(tamanho)+(coluna)])
/// calcula o menor de 2 números
#define menor(a,b)  (((a)<(b))? (a):(b))

/// número de elementos máximo a serem impressos (linha e coluna)
#define MAX_PRINT   7

/// erro máximo aceitável
#define ERRO_MAXIMO 0.1

// número default de threads para versão com threads no host
#define DEFAULT_NTHREADS 16

// Constantes booleanas
#define FALSE 0
#define TRUE -1

//
// tipos de dados
//

/// @brief Tipo de dado utilizado para os elementos da matriz e do vetor.
typedef float data_t;

/// @brief Estrutura de argumentos para a thread
typedef struct {
    int threadId;       // id da thread
    data_t *hmA;        // matriz A
    data_t *hvB;        // vetor B
    int nIncognitas;    // número de incógnitas (tamanho da matriz)
    int passo;          // passo atual
} threadArgs_t;

/// @brief Estrutura de retorno da thread 
typedef struct {
    int status;
} threadReturn_t;

//
// Macros
//

/// Macro para calcular a diferença de tempo em milissegundos
#define timedifference_msec(start, stop) ((double)(stop - start) * 1000.0 / CLOCKS_PER_SEC)

/// Macro para calcular a diferença de tempo em milissegundos utilizando timespec
#define timedifference_msec_spec(start, stop) ((double)((stop).tv_sec - (start).tv_sec) * 1000.0 + (double)((stop).tv_nsec - (start).tv_nsec) / 1000000.0)

//
// Protótipos
//
void exibeMatriz(data_t *hmA, int nIncognitas);
void exibeVetor(data_t *hvB, int nIncognitas);
data_t *leMatriz(char *nome, int nIncognitas);
data_t *leVetor(char *nome, int nIncognitas);
data_t *calculaX(data_t *hmA, data_t *hvB, int nIncognitas);
void verificaResultado(data_t *hmA, data_t *hvX, data_t *hvB, int nIncognitas);
void exibeLinhaMatriz(data_t *hmA, int linha, int nIncognitas);
void processaVetoresThread(data_t *hmA, data_t *hvB, int nIncognitas);
void processaVetoresGPU(data_t *hmA, data_t *hvB, int nIncognitas);
void parseArguments(int argc, char *argv[]);
void uso(char *nome);

// variáveis globais
extern int nIncognitas;
extern char *filenameMatrizA;
extern char *filenameVetorB;
extern char *filenameTempos;
extern int threadsPerBlock;
extern int blocksPerGrid;
extern int maxPrint;
extern int nThreads;
#endif