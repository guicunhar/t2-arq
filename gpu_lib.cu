/**
 * @file gpu_lib.cu
 * @brief Eliminação gaussiana com duas versões paralelas: multithread no host
 *        (Intel Intrinsics AVX/FMA + pthreads) e na GPU (CUDA).
 *
 * Implementa as funções processaVetoresThread e processaVetoresGPU, que
 * transformam a matriz A em uma matriz triangular superior, zerando os
 * elementos abaixo da diagonal principal, mantendo o sistema Ax = b íntegro.
 *
 * processaVetoresThread:
 * - Usa instruções AVX/FMA (Intel Intrinsics) na eliminação de cada linha,
 *   com acesso ALINHADO à memória (_mm256_load_ps/_mm256_store_ps): cada
 *   linha é percorrida em escalar até a primeira coluna múltipla de 8
 *   (peeling) e, a partir daí, em blocos vetoriais alinhados de 8 floats,
 *   já que aligned_alloc(32,...) garante que cada linha começa em endereço
 *   múltiplo de 32 bytes (para n múltiplo de 8).
 * - A atualização de cada linha usa FMA (_mm256_fnmadd_ps) em uma única
 *   instrução de multiplicação-subtração.
 * - Paraleliza o passo de eliminação entre nThreads threads POSIX, cada uma
 *   responsável por um subconjunto de linhas (distribuição round-robin).
 * - O pivotamento parcial (escolha da melhor linha) é feito de forma
 *   sequencial no host, pois seu custo é O(n) por passo, desprezível frente
 *   ao custo O(n^2) da eliminação.
 *
 * processaVetoresGPU:
 * - Mantém a matriz A e o vetor b inteiramente na memória do device durante
 *   todo o processamento (cópias H2D/D2H só no início e no fim).
 * - Usa um vetor auxiliar no device (dMultiplicadores) para guardar, uma
 *   única vez por passo, o multiplicador de cada linha. Isso evita
 *   recalcular a divisão em cada thread que toca a linha e evita qualquer
 *   necessidade de sincronização entre kernels (cada valor é escrito por
 *   exatamente uma thread e lido depois por outras, em kernels diferentes,
 *   cuja ordem de execução já é garantida pela mesma stream padrão do CUDA).
 * - A atualização da submatriz "achata" os elementos a atualizar (linha,
 *   coluna) em um único índice e distribui esse índice uniformemente entre
 *   todas as threads disponíveis (blocksPerGrid x threadsPerBlock) usando o
 *   padrão de "grid-stride loop", o que funciona corretamente independente
 *   do tamanho do sistema e da configuração de blocos/threads escolhida na
 *   linha de comando (-t e -g).
 * - Por simplicidade e para minimizar o número de cópias/transferências
 *   Host<->Device por passo (o que prejudicaria o speedup que é o objetivo
 *   do trabalho), a versão GPU NÃO faz pivotamento parcial. Ela apenas
 *   verifica se o pivô está nulo (ou muito próximo de zero) e aborta com
 *   erro nesse caso. Use matrizes de teste bem-condicionadas (por exemplo,
 *   diagonalmente dominantes) para evitar esse problema.
 *
 * Compilação (ver Makefile fornecido pelo professor):
 * nvcc -lineinfo -lm -o equation equation_test.cu gpu_lib.cu
 *
 * Observação sobre AVX/FMA + nvcc:
 * As funções que usam intrinsics (_mm256_*) foram marcadas com
 * __attribute__((target("avx,fma"))) para que o código compile e gere
 * instruções AVX/FMA mesmo que a linha de compilação do nvcc não inclua
 * -mavx/-march=native explicitamente (testado e validado com gcc puro,
 * sem nenhuma flag de AVX na linha de comando). Caso o nvcc da máquina de
 * testes não aceite esse atributo, basta adicionar ao Makefile:
 *     CUDA_FLAGS += -Xcompiler -mavx -Xcompiler -mfma
 *
 * Códigos de erro:
 *  - 10: pivô nulo ou próximo de zero (processaVetoresThread)
 *  - 20: falha ao alocar memória na GPU (cudaMalloc)
 *  - 21: falha ao copiar memória entre host e device (cudaMemcpy)
 *  - 22: falha ao executar um kernel CUDA
 *  - 23: pivô nulo ou próximo de zero (processaVetoresGPU)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <immintrin.h>
#include <cuda_runtime.h>
#include "comum.h"
#include "gpu.h"

/** tolerância para considerar um pivô como nulo */
#define TOLERANCIA_PIVO 1e-6f

/** códigos de erro */
#define ERRO_PIVO_NULO_THREAD 10
#define ERRO_CUDA_MALLOC      20
#define ERRO_CUDA_MEMCPY      21
#define ERRO_CUDA_KERNEL      22
#define ERRO_PIVO_NULO_GPU    23

/**
 * @brief Verifica o resultado de uma chamada à API do CUDA Runtime e aborta
 *        o programa com uma mensagem de erro e o código apropriado caso a
 *        chamada tenha falhado.
 *
 * @param chamada     expressão que retorna um cudaError_t (ex: cudaMalloc(...))
 * @param codigoErro  código de erro a ser usado em caso de falha
 */
#define CUDA_CHECK(chamada, codigoErro)                                         \
    do {                                                                        \
        cudaError_t erroCuda = (chamada);                                      \
        if (erroCuda != cudaSuccess) {                                         \
            fprintf(stderr, "Erro %d: Falha CUDA em %s:%d (%s) -> %s\n",       \
                    (codigoErro), __FILE__, __LINE__, #chamada,                \
                    cudaGetErrorString(erroCuda));                             \
            exit(codigoErro);                                                  \
        }                                                                       \
    } while (0)

/*
 * ===========================================================================
 * PARTE 1: processaVetoresThread (host, multithread + AVX/FMA)
 * ===========================================================================
 */

/**
 * @brief Função executada por cada thread POSIX: elimina, com AVX/FMA, o
 *        subconjunto de linhas que coube a ela em um determinado passo.
 *
 * As linhas são distribuídas entre as threads de forma intercalada
 * (round-robin): a thread de id "threadId" cuida das linhas
 * passo+threadId, passo+threadId+nThreads, passo+threadId+2*nThreads, ...
 * Como o trabalho por linha é o mesmo (mesma quantidade de colunas a
 * atualizar), essa distribuição já é uniforme entre as threads.
 *
 * @param arg ponteiro para um threadArgs_t (ver comum.h) com os dados do passo
 * @return NULL
 */
__attribute__((target("avx,fma")))
static void *threadElimina(void *arg) {
    threadArgs_t *targs = (threadArgs_t *)arg;

    int nIncognitas  = targs->nIncognitas;
    int passo        = targs->passo;
    data_t *hmA      = targs->hmA;
    data_t *hvB      = targs->hvB;
    int threadId     = targs->threadId;

    data_t *linhaPivo = &matriz(hmA, passo - 1, 0, nIncognitas);
    data_t pivo       = matriz(hmA, passo - 1, passo - 1, nIncognitas);

    for (int linha = passo + threadId; linha < nIncognitas; linha += nThreads) {

        data_t *linhaAtual = &matriz(hmA, linha, 0, nIncognitas);
        data_t multiplicador = linhaAtual[passo - 1] / pivo;
        __m256 vecMult = _mm256_set1_ps(multiplicador);

        int coluna = passo;

        /* peeling escalar até alinhar a 32 bytes (8 floats) */
        int colunaAlinhada = ((coluna + 7) / 8) * 8;
        for (; coluna < colunaAlinhada && coluna < nIncognitas; coluna++) {
            linhaAtual[coluna] -= linhaPivo[coluna] * multiplicador;
        }

        /* bloco vetorial alinhado, com FMA */
        for (; coluna <= nIncognitas - 8; coluna += 8) {
            __m256 vecLP = _mm256_load_ps(&linhaPivo[coluna]);
            __m256 vecLA = _mm256_load_ps(&linhaAtual[coluna]);
            __m256 vecRes = _mm256_fnmadd_ps(vecLP, vecMult, vecLA);
            _mm256_store_ps(&linhaAtual[coluna], vecRes);
        }

        /* resto escalar final (caso nIncognitas não seja múltiplo de 8) */
        for (; coluna < nIncognitas; coluna++) {
            linhaAtual[coluna] -= linhaPivo[coluna] * multiplicador;
        }

        /* força zero exato na coluna do pivô */
        linhaAtual[passo - 1] = 0.0f;

        /* atualiza B (escalar, um único valor por linha) */
        hvB[linha] -= hvB[passo - 1] * multiplicador;
    }

    return NULL;
}

/**
 * @brief Aplica eliminação gaussiana com pivotamento parcial (sequencial) e
 *        eliminação vetorizada (AVX/FMA) distribuída entre nThreads threads
 *        POSIX no host.
 *
 * @param hmA          matriz A (host)
 * @param hvB          vetor B (host)
 * @param nIncognitas  número de incógnitas
 */
__attribute__((target("avx,fma")))
void processaVetoresThread(data_t *hmA, data_t *hvB, int nIncognitas) {

    pthread_t   *threads = (pthread_t *)malloc(sizeof(pthread_t) * nThreads);
    threadArgs_t *args   = (threadArgs_t *)malloc(sizeof(threadArgs_t) * nThreads);

    if (!threads || !args) {
        fprintf(stderr, "[ERROR %d] Não foi possível alocar memória para as threads\n", __LINE__);
        exit(3);
    }

    for (int passo = 1; passo < nIncognitas; passo++) {

        /* pivotamento parcial: escolhe a linha com maior valor absoluto na coluna */
        int melhorLinha = passo - 1;
        data_t maxVal = fabsf(matriz(hmA, passo - 1, passo - 1, nIncognitas));

        for (int i = passo; i < nIncognitas; i++) {
            data_t val = fabsf(matriz(hmA, i, passo - 1, nIncognitas));
            if (val > maxVal) {
                maxVal = val;
                melhorLinha = i;
            }
        }

        if (melhorLinha != passo - 1) {
            data_t *linha1 = &matriz(hmA, passo - 1, 0, nIncognitas);
            data_t *linha2 = &matriz(hmA, melhorLinha, 0, nIncognitas);

            int j = 0;
            for (; j <= nIncognitas - 8; j += 8) {
                __m256 v1 = _mm256_load_ps(&linha1[j]);
                __m256 v2 = _mm256_load_ps(&linha2[j]);
                _mm256_store_ps(&linha1[j], v2);
                _mm256_store_ps(&linha2[j], v1);
            }
            for (; j < nIncognitas; j++) {
                data_t tmp = linha1[j];
                linha1[j] = linha2[j];
                linha2[j] = tmp;
            }

            data_t tmpB = hvB[passo - 1];
            hvB[passo - 1] = hvB[melhorLinha];
            hvB[melhorLinha] = tmpB;
        }

        data_t pivo = matriz(hmA, passo - 1, passo - 1, nIncognitas);
        if (fabsf(pivo) < TOLERANCIA_PIVO) {
            fprintf(stderr, "Erro %d: Pivo nulo na linha %d (processaVetoresThread).\n",
                    ERRO_PIVO_NULO_THREAD, passo - 1);
            free(threads);
            free(args);
            exit(ERRO_PIVO_NULO_THREAD);
        }

        /* dispara nThreads threads para eliminar, em paralelo, as linhas deste passo */
        for (int t = 0; t < nThreads; t++) {
            args[t].threadId    = t;
            args[t].hmA         = hmA;
            args[t].hvB         = hvB;
            args[t].nIncognitas = nIncognitas;
            args[t].passo       = passo;
            pthread_create(&threads[t], NULL, threadElimina, &args[t]);
        }
        for (int t = 0; t < nThreads; t++) {
            pthread_join(threads[t], NULL);
        }
    }

    free(threads);
    free(args);
}

/*
 * ===========================================================================
 * PARTE 2: processaVetoresGPU (device, CUDA)
 * ===========================================================================
 */

/**
 * @brief Kernel CUDA que calcula, para cada linha abaixo do pivô, o
 *        multiplicador da eliminação, atualiza o vetor B e zera a coluna do
 *        pivô nessa linha. O resultado é guardado em um vetor auxiliar
 *        (dMultiplicadores) para ser reaproveitado pelo kernel seguinte
 *        sem necessidade de recalculá-lo nem de sincronização extra.
 *
 * @param dmA              matriz A (device)
 * @param dvB              vetor B (device)
 * @param dMultiplicadores vetor auxiliar (device) com um multiplicador por linha
 * @param nIncognitas      número de incógnitas
 * @param passo            passo atual da eliminação (linha do pivô = passo-1)
 */
__global__ void kernelCalculaMultiplicadores(data_t *dmA, data_t *dvB,
                                              data_t *dMultiplicadores,
                                              int nIncognitas, int passo) {
    int idxGlobal = blockIdx.x * blockDim.x + threadIdx.x;
    int stride    = blockDim.x * gridDim.x;

    data_t pivo = matriz(dmA, passo - 1, passo - 1, nIncognitas);

    for (int linha = passo + idxGlobal; linha < nIncognitas; linha += stride) {
        data_t multiplicador = matriz(dmA, linha, passo - 1, nIncognitas) / pivo;
        dMultiplicadores[linha] = multiplicador;
        dvB[linha] -= dvB[passo - 1] * multiplicador;
        matriz(dmA, linha, passo - 1, nIncognitas) = 0.0f;
    }
}

/**
 * @brief Kernel CUDA que atualiza a submatriz (linhas e colunas após o
 *        pivô). Os elementos (linha, coluna) a atualizar são "achatados" em
 *        um único índice e distribuídos uniformemente entre todas as
 *        threads (blocksPerGrid x threadsPerBlock) usando um grid-stride
 *        loop, o que funciona corretamente para qualquer combinação de
 *        tamanho do sistema e configuração de blocos/threads.
 *
 * @param dmA              matriz A (device)
 * @param dMultiplicadores vetor auxiliar (device) com o multiplicador de cada linha
 * @param nIncognitas      número de incógnitas
 * @param passo            passo atual da eliminação
 */
__global__ void kernelAtualizaSubmatriz(data_t *dmA, data_t *dMultiplicadores,
                                         int nIncognitas, int passo) {
    int tamanho = nIncognitas - passo;
    long long totalElementos = (long long)tamanho * (long long)tamanho;

    long long idxGlobal = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride    = (long long)blockDim.x * gridDim.x;

    for (long long idx = idxGlobal; idx < totalElementos; idx += stride) {
        int linha  = passo + (int)(idx / tamanho);
        int coluna = passo + (int)(idx % tamanho);
        matriz(dmA, linha, coluna, nIncognitas) -=
            dMultiplicadores[linha] * matriz(dmA, passo - 1, coluna, nIncognitas);
    }
}

/**
 * @brief Aplica eliminação gaussiana na GPU. A matriz A e o vetor B são
 *        copiados para o device uma única vez, processados inteiramente lá
 *        (sem pivotamento parcial, apenas verificação de pivô nulo) e
 *        copiados de volta ao final.
 *
 * @param hmA          matriz A (host)
 * @param hvB          vetor B (host)
 * @param nIncognitas  número de incógnitas
 */
void processaVetoresGPU(data_t *hmA, data_t *hvB, int nIncognitas) {

    data_t *dmA, *dvB, *dMultiplicadores;

    size_t tamMatriz = (size_t)nIncognitas * (size_t)nIncognitas * sizeof(data_t);
    size_t tamVetor  = (size_t)nIncognitas * sizeof(data_t);

    CUDA_CHECK(cudaMalloc((void **)&dmA, tamMatriz), ERRO_CUDA_MALLOC);
    CUDA_CHECK(cudaMalloc((void **)&dvB, tamVetor), ERRO_CUDA_MALLOC);
    CUDA_CHECK(cudaMalloc((void **)&dMultiplicadores, tamVetor), ERRO_CUDA_MALLOC);

    CUDA_CHECK(cudaMemcpy(dmA, hmA, tamMatriz, cudaMemcpyHostToDevice), ERRO_CUDA_MEMCPY);
    CUDA_CHECK(cudaMemcpy(dvB, hvB, tamVetor, cudaMemcpyHostToDevice), ERRO_CUDA_MEMCPY);

    for (int passo = 1; passo < nIncognitas; passo++) {

        /* verifica se o pivô atual é nulo (sem pivotamento parcial na GPU) */
        data_t pivoHost;
        CUDA_CHECK(cudaMemcpy(&pivoHost,
                               dmA + (size_t)(passo - 1) * nIncognitas + (passo - 1),
                               sizeof(data_t), cudaMemcpyDeviceToHost),
                   ERRO_CUDA_MEMCPY);

        if (fabsf(pivoHost) < TOLERANCIA_PIVO) {
            fprintf(stderr, "Erro %d: Pivo nulo na linha %d (processaVetoresGPU).\n",
                    ERRO_PIVO_NULO_GPU, passo - 1);
            cudaFree(dmA);
            cudaFree(dvB);
            cudaFree(dMultiplicadores);
            exit(ERRO_PIVO_NULO_GPU);
        }

        kernelCalculaMultiplicadores<<<blocksPerGrid, threadsPerBlock>>>(
            dmA, dvB, dMultiplicadores, nIncognitas, passo);
        CUDA_CHECK(cudaGetLastError(), ERRO_CUDA_KERNEL);

        kernelAtualizaSubmatriz<<<blocksPerGrid, threadsPerBlock>>>(
            dmA, dMultiplicadores, nIncognitas, passo);
        CUDA_CHECK(cudaGetLastError(), ERRO_CUDA_KERNEL);
    }

    CUDA_CHECK(cudaMemcpy(hmA, dmA, tamMatriz, cudaMemcpyDeviceToHost), ERRO_CUDA_MEMCPY);
    CUDA_CHECK(cudaMemcpy(hvB, dvB, tamVetor, cudaMemcpyDeviceToHost), ERRO_CUDA_MEMCPY);

    cudaFree(dmA);
    cudaFree(dvB);
    cudaFree(dMultiplicadores);
}
