/**
 * Programa de teste para o trabalho 2 de GPU, que resolve um sistema de equações lineares usando eliminação gaussiana.
 * O programa lê uma matriz A e um vetor B de arquivos binários, 
 * processa os vetores (transforma a matriz A e o vetor B) usando uma versão multithread no host e outra na GPU, 
 * calcula o vetor X usando a matriz A transformada e o vetor B transformado, 
 * e verifica o resultado comparando o produto A*X com o vetor B original.
 * 
 * @see Enunciado do trabalho    
 */
//#define DEBUG
#define FLOAT 4
#define DOUBLE 8
#define PRECISION_TYPE FLOAT

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <unistd.h>
//#include <immintrin.h>

#include "comum.h"
#include "gpu.h"

/*
 * Variáveis globais
 */
// Número de incógnitas do sistema (igual ao número de linhas ou colunas da matriz A ou tamanho do vetor B)
int nIncognitas;
// Nome do arquivo contendo a matriz A
char *filenameMatrizA;
// Nome do arquivo contendo o vetor B
char *filenameVetorB;
// Nome do arquivo com a saída de dados
char *filenameTempos = (char *)"tempos.csv";
// Quantidade de threads por bloco
int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
// Quantidade de blocos por grid
int blocksPerGrid = DEFAULT_MAX_BLOCKS_PER_GRID;
// Quantidade de threads (para a versão com threads no host)
int nThreads = DEFAULT_NTHREADS;
// Número máximo de valores ao exibir matrizes ou vetores
int maxPrint = MAX_PRINT;

/**
 * Função main
 * Realiza o processamento do sistema de equações lineares, 
 * usando tanto uma versão multithread no host quanto uma versão na GPU, 
 * e verifica os resultados.
 * Implementar as funções
 * - processaVetoresThread: versão multithread no host
 * - processaVetoresGPU: versão na GPU
 * 
 * @author Alexandre Meslin
 * @param argc  número de argumentos
 * @param argv  vetor com os argumentos
 * @return  código de saída do programa
 * @see enunciado do trabalho
 */
int main(int argc, char *argv[]) {
    // Ax = B
    data_t *hmA, *hvB, *hvX;
    FILE *arq;

    // medida de tempo
    struct timespec start_time_spec, stop_time_spec;

    parseArguments(argc, argv);

    // Lê os dados dos arquivos
    // Matriz A
    hmA = leMatriz(filenameMatrizA, nIncognitas);
    hvB = leVetor(filenameVetorB, nIncognitas);

#ifdef DEBUG    
    fprintf(stderr, "[INFO %3d] Vetores lidos dos arquivos:\n", __LINE__);
    exibeMatriz(hmA, nIncognitas);
    exibeVetor(hvB, nIncognitas);
#endif

    /*
     * Processamento multithread no host
     */
    // liga o cronômetro
    clock_gettime(CLOCK_MONOTONIC, &start_time_spec);
    // Processa os vetores (transforma a matriz A e o vetor B)
    processaVetoresThread(hmA, hvB, nIncognitas);
    // desliga o cronômetro
    clock_gettime(CLOCK_MONOTONIC, &stop_time_spec);
    // Show init exec time
    if(!(arq = fopen("tempos.csv", "a"))) {
        fprintf(stderr, "[ERROR %d] Não foi possível abrir o arquivo de tempos.csv para incluir um novo tempo\n", __LINE__);
        exit(15);
    }
    fprintf(arq, "%s,%f,", argv[0], timedifference_msec_spec(start_time_spec, stop_time_spec));
    fprintf(stderr, "[INFO %3d] Tempo do cálculo no device: %f ms\n", __LINE__, timedifference_msec_spec(start_time_spec, stop_time_spec));
    fclose(arq);

#ifdef DEBUG    
    fprintf(stderr, "[INFO %3d] Vetores transformados:\n", __LINE__);
    exibeMatriz(hmA, nIncognitas);
    exibeVetor(hvB, nIncognitas);
#endif
    hvX = calculaX(hmA, hvB, nIncognitas);

    // verifica o resultado
    free(hmA);
    free(hvB);
    hmA = leMatriz(filenameMatrizA, nIncognitas);
    hvB = leVetor(filenameVetorB, nIncognitas);
    verificaResultado(hmA, hvX, hvB, nIncognitas);

    /*
     * Processamento multithread no device
     */
    // liga o cronômetro
    clock_gettime(CLOCK_MONOTONIC, &start_time_spec);
    // Processa os vetores (transforma a matriz A e o vetor B)
    processaVetoresGPU(hmA, hvB, nIncognitas);
    // desliga o cronômetro
    clock_gettime(CLOCK_MONOTONIC, &stop_time_spec);
    // Show init exec time
    if(!(arq = fopen("tempos.csv", "a"))) {
        fprintf(stderr, "[ERROR %d] Não foi possível abrir o arquivo de tempos.csv para incluir um novo tempo\n", __LINE__);
        exit(15);
    }
    fprintf(arq, "%s,%f,", argv[0], timedifference_msec_spec(start_time_spec, stop_time_spec));
    fprintf(stderr, "[INFO %3d] Tempo do cálculo na GPU: %f ms\n", __LINE__, timedifference_msec_spec(start_time_spec, stop_time_spec));
    fclose(arq);

#ifdef DEBUG    
    fprintf(stderr, "[INFO %3d] Vetores transformados:\n", __LINE__);
    exibeMatriz(hmA, nIncognitas);
    exibeVetor(hvB, nIncognitas);
#endif
    hvX = calculaX(hmA, hvB, nIncognitas);

    // verifica o resultado
    free(hmA);
    free(hvB);
    hmA = leMatriz(filenameMatrizA, nIncognitas);
    hvB = leVetor(filenameVetorB, nIncognitas);
    verificaResultado(hmA, hvX, hvB, nIncognitas);

    // Libera memória
    free(hmA);
    free(hvB);
    free(hvX);

    return 0;
}

/**
 * Função parseArguments: realiza o parse de argumentos da linha de comando.
 * Argumentos válidos:
 * -m <nome do arquivo com a matriz A>
 * -n <quantidade de incógnitas>
 * -o <nome do arquivo de saída de tempos> (opcional - default tempos.csv)
 * -v <nome do arquivo com o vetor B>
 * -t <número de threads> (opcional - default DEFAULT_NTHREADS)
 * 
 * @param argc  número de argumentos
 * @param argv  vetor com os argumentos
*/
void parseArguments(int argc, char *argv[]) {
    opterr = 0;
    int opcao;      // opção selecionada
    char *shortOpts = (char *)"g:m:n:o:p:T:t:v:"; // opções válidas
    int flagFilenameMatrizA = FALSE;
    int flagFilenameVetorB = FALSE;
    int flagNIncognitas = FALSE;

    while((opcao=getopt(argc, argv, shortOpts)) != -1) {
        switch (opcao) {
        case 'g':
            threadsPerBlock = atoi(optarg);
            break;

        case 'm':
            filenameMatrizA = optarg;
            flagFilenameMatrizA = TRUE;
            break;

        case 'n':
            nIncognitas = atoi(optarg);
            flagNIncognitas = TRUE;
            break;

        case 'o':            // opcional
            filenameTempos = optarg;
            break;

        case 'p':
            maxPrint = atoi(optarg);
            break;

        case 't':
            blocksPerGrid = atoi(optarg);
            break;

        case 'T':
            nThreads = atoi(optarg);
            break;

        case 'v':
            filenameVetorB = optarg;
            flagFilenameVetorB = TRUE;
            break;
        
        default:
            fprintf(stderr, "[ERROR %d] Opção desconhecida: %c\n", __LINE__, optopt);
            break;
        }
    }
    if(!flagFilenameMatrizA) fprintf(stderr, "[ERROR %d] Nome do arquivo com a matriz A faltando (opção -m)\n", __LINE__);
    if(!flagNIncognitas)     fprintf(stderr, "[ERROR %d] Número de incógnitas do sistema faltando (opção -n)\n", __LINE__);
    if(!flagFilenameVetorB)  fprintf(stderr, "[ERROR %d] Nome do arquivo com o vetor B faltando (opção -v)\n", __LINE__);
    if(!flagFilenameMatrizA || !flagFilenameVetorB || !flagNIncognitas) {
        uso(argv[0]);
        exit(-1);
    }
}

void uso(char *nome) {
    fprintf(stderr, "Uso: %s -m <arquivo com a matriz A> -n <número de incógnitas do sistema> [-o <nome do arquivo de tempos>] -v <nome do arquivo com o vetor B>\n", nome);
}

void exibeMatriz(data_t *hmA, int nIncognitas) {
    fprintf(stderr, "[DEBUG %d] ============================================\n", __LINE__);
    for(int linha=0; linha<nIncognitas; linha++) {
        if(linha > maxPrint) {
            fprintf(stderr, "...\n");
            exibeLinhaMatriz(hmA, nIncognitas-1, nIncognitas);
            fputc('\n', stderr);
            break;
        }
        exibeLinhaMatriz(hmA, linha, nIncognitas);
        fputc('\n', stderr);
    }
    fprintf(stderr, "============================================\n");
}

void exibeLinhaMatriz(data_t *hmA, int linha, int nIncognitas) {
    fprintf(stderr, "Linha %d: ", linha);
    for(int coluna=0; coluna<nIncognitas; coluna++) {
        if(coluna > maxPrint) {
            fprintf(stderr, "... %7.4f", matriz(hmA, linha, nIncognitas-1, nIncognitas));
            break;
        }
        fprintf(stderr, "%7.4f ", matriz(hmA, linha, coluna, nIncognitas));
    }
}

void exibeVetor(data_t *hvB, int nIncognitas) {
    fprintf(stderr, "[DEBUG %d] Vetor B =\n", __LINE__);
    fprintf(stderr, "============================================\n");
    for(int linha=0; linha<nIncognitas; linha++) {
        if(linha > maxPrint) {
            fprintf(stderr, "... %7.4f", hvB[nIncognitas -1]);
            break;
        }
        fprintf(stderr, "%7.4f ", hvB[linha]);
    }
    fputc('\n', stderr);
    fprintf(stderr, "============================================\n");
}

/**
 * Função leMatriz: lê uma matriz de um arquivo binário.<br>
 * O arquivo deve conter nIncognitas*nIncognitas elementos do tipo data_t, 
 * armazenados em formato binário.
 * 
 * @param nome  nome do arquivo a ser lido
 * @param nIncognitas  número de incógnitas (tamanho da matriz)
 * @return  ponteiro para a matriz lida
 */
data_t *leMatriz(char *nome, int nIncognitas) {
    data_t *hmA;
    FILE *arq;
    int qtd;

    if(!(hmA = (data_t *)aligned_alloc(32, sizeof(data_t) * nIncognitas * nIncognitas))) {
        fprintf(stderr, "[ERROR %d] Não foi possível alocar memória para a matriz A[%d, %3d]\n", __LINE__, nIncognitas, nIncognitas);
        exit(1);
    }
    if(!(arq = fopen(nome, "rb"))) {
        fprintf(stderr, "[ERROR %d] Não foi possível abrir o arquivo %s para leitura\n", __LINE__, nome);
        exit(6);
    }
    if((qtd = fread(hmA, sizeof(data_t), nIncognitas*nIncognitas, arq)) != nIncognitas*nIncognitas) {
        fprintf(stderr, "[ERROR %d] Tamanho de arquivo incompatível (%d != %d)\n", __LINE__, qtd, nIncognitas*nIncognitas);
        exit(7);
    }
    fclose(arq);
    return hmA;
}

data_t *leVetor(char *nome, int nIncognitas) {
    data_t *hvB;
    FILE *arq;
    int qtd;

    if(!(hvB = (data_t *)aligned_alloc(32, sizeof(data_t) * nIncognitas))) {
        fprintf(stderr, "[ERROR %d] Não foi possível alocar memória para o vetor B[%3d]\n", __LINE__, nIncognitas);
        exit(2);
    }
    if(!(arq = fopen(nome, "rb"))) {
        fprintf(stderr, "[ERROR %d] Não foi possível abrir o arquivo %s para leitura\n", __LINE__, nome);
        exit(4);
    }
    if((qtd = fread(hvB, sizeof(data_t), nIncognitas, arq)) != nIncognitas) {
        fprintf(stderr, "[ERROR %d] Tamanho de arquivo incompatível (%d != %d)\n", __LINE__, qtd, nIncognitas);
        exit(5);
    }
    fclose(arq);
    return hvB;
}

data_t *calculaX(data_t *hmA, data_t *hvB, int nIncognitas) {
    data_t *hvX;

    // cria espaço em memória para o vetor X
    if(!(hvX = (data_t *)aligned_alloc(32, sizeof(data_t) * nIncognitas))) {
        fprintf(stderr, "[ERROR %d] Não foi possível alocar memória para o vetor X[%3d]\n", __LINE__, nIncognitas);
        exit(2);
    }

    // Calcula o vetor X
    for(int diagonal=nIncognitas-1; diagonal>=0; diagonal--) {
        data_t somatorio = 0;
        for(int coluna=nIncognitas-1; coluna>diagonal; coluna--) {
            somatorio += hvX[coluna] * matriz(hmA, diagonal, coluna, nIncognitas);
            if(isnan(somatorio)) {
                fprintf(stderr, "[ERROR %d] Somatório virou NaN em %d,%d (%f %f)\n", __LINE__, diagonal, coluna, hvX[coluna], matriz(hmA, diagonal, coluna, nIncognitas));
                exit(-1);
            }
        }
        hvX[diagonal] = (hvB[diagonal] - somatorio) / matriz(hmA, diagonal, diagonal, nIncognitas);
    }

#ifdef DEBUG    
    // Exibe o resultado
    fprintf(stderr, "[DEBUG %3d] X =\n", __LINE__);
    exibeVetor(hvX, nIncognitas);
#endif

    return hvX;
}

void verificaResultado(data_t *hmA, data_t *hvX, data_t *hvB, int nIncognitas) {
    printf("[INFO %d] Verificando resultado... ", __LINE__);
    for(int linha=0; linha<nIncognitas; linha++) {
        data_t somatorio=0;
        for(int coluna=0; coluna<nIncognitas; coluna++) {
            somatorio += matriz(hmA, linha, coluna, nIncognitas) * hvX[coluna];
        }
        if(fabs(somatorio - hvB[linha]) > fabs(somatorio) * 0.01 || fabs(somatorio - hvB[linha]) > hvB[linha] * 0.1 || isnan(somatorio)) {
            fprintf(stderr, "[ERROR %3d] Resultado errado na linha %d (%f != %f)\n", __LINE__, linha, somatorio, hvB[linha]);
#if PRECISION_TYPE == DOUBLE
            // Somente deve abortar se a precisão for double, porque para float o erro pode ser maior devido à menor precisão. 
            fprintf(stderr, "[ERROR %3d] Abortando...\n", __LINE__);
            exit(8);
#else
            return; // para float, apenas retorna e não aborta, porque o erro pode ser maior devido à menor precisão.
#endif
        }
    }
    printf("Ok\n");
}

void processaVetoresEscalarmente(data_t *hmA, data_t *hvB, int nIncognitas) {
    // Transforma a matriz A e o vetor B
    for(int passo=1; passo<nIncognitas; passo++) {
        for(int linha=passo; linha<nIncognitas; linha++) {
            data_t multiplicador = matriz(hmA, linha, passo-1, nIncognitas) / matriz(hmA, passo-1, passo-1, nIncognitas);
            for(int coluna=passo-1; coluna<nIncognitas; coluna++) {
                matriz(hmA, linha, coluna, nIncognitas) -= matriz(hmA, passo-1, coluna, nIncognitas) * multiplicador;
            }
            hvB[linha] -= hvB[passo-1] * multiplicador;
        }
    } 
}
