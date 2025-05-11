#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

#define BLOCK_SIZE 32
// Uncomment for ISPC
// #include "module_ispc.h"
// using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX)
{
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX) + y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val)
{
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b,
                         const int &sizeX, const int &sizeY, const int &sizeZ)
{
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b,
                         const int &sizeX, const int &sizeY, const int &sizeZ, float &val)
{
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor)
{
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 *
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                               int B, int H, int N, int d)
{

    // Q, K, V are passed in with Shape: (B, H, N, d)
    // QK^t Intermediate Tensor has Shape (N, N)

    // Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    // Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
           for (int j = 0; j < N; j++) {
               float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
               twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */

    // -------- YOUR CODE HERE  -------- //

    // loop over Batch Size
    for (int b = 0; b < B; b++)
    {

        // loop over Heads
        for (int h = 0; h < H; h++)
        {

            // loop over Sequence Length (rows)
            for (int i = 0; i < N; i++)
            {
                // loop over Embedding Dimensionality (rows of transposed)
                for (int j = 0; j < N; j++)
                {
                    float valQKT = 0.0;
                    // loop over cols
                    for (int k = 0; k < d; k++)
                    {
                        float valQ = fourDimRead(Q, b, h, i, k, H, N, d);
                        float valK = fourDimRead(K, b, h, j, k, H, N, d);
                        valQKT += valQ * valK;
                    }
                    twoDimWrite(QK_t, i, j, N, valQKT);
                }
            }
            // part b
            for (int i = 0; i < N; i++)
            {
                float sumExp = 0.0;
                float *Exparr = new float[N];

                for (int j = 0; j < N; j++)
                {
                    float val = twoDimRead(QK_t, i, j, N);
                    val = exp(val);
                    Exparr[j] = val;
                    sumExp += val;
                }
                for (int j = 0; j < N; j++)
                {
                    float val = Exparr[j] / sumExp;
                    twoDimWrite(QK_t, i, j, N, val);
                }
                if (Exparr)
                {
                    delete[] Exparr;
                }
            }

            // part c
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    float val = 0.0;
                    for (int k = 0; k < N; k++)
                    {
                        float qkt = twoDimRead(QK_t, i, k, N);
                        float v = fourDimRead(V, b, h, k, j, H, N, d);
                        val += qkt * v;
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, val);
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                                        int B, int H, int N, int d)
{

    // Q, K, V are passed in with Shape: (B, H, N, d)
    // QK^t Intermediate Tensor has Shape (N, N)

    // Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    // Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //

    // loop over Batch Size
    for (int b = 0; b < B; b++)
    {

        // loop over Heads
        for (int h = 0; h < H; h++)
        {

            // loop over Sequence Length (rows)
            for (int i = 0; i < N; i += BLOCK_SIZE)
            {
                // loop over Embedding Dimensionality (rows of transposed)
                for (int j = 0; j < N; j += BLOCK_SIZE)
                {
                    int minVali = N < i + BLOCK_SIZE ? N : i + BLOCK_SIZE;
                    for (int blockRow = i; blockRow < minVali; blockRow++)
                    {
                        int minValj = N < j + BLOCK_SIZE ? N : j + BLOCK_SIZE;
                        for (int blockCol = j; blockCol < minValj; blockCol++)
                        {

                            float valQKT = 0.0;
                            // loop over cols
                            for (int k = 0; k < d; k++)
                            {
                                float valQ = fourDimRead(Q, b, h, blockRow, k, H, N, d);
                                float valK = fourDimRead(K, b, h, blockCol, k, H, N, d);
                                valQKT += valQ * valK;
                            }
                            twoDimWrite(QK_t, blockRow, blockCol, N, valQKT);
                        }
                    }
                }
            }
            // part b
            for (int i = 0; i < N; i++)
            {
                float sumExp = 0.0;
                std::vector<float> Exparr(N);

                for (int j = 0; j < N; j++)
                {
                    float val = twoDimRead(QK_t, i, j, N);
                    val = exp(val);
                    Exparr[j] = val;
                    sumExp += val;
                }
                for (int j = 0; j < N; j++)
                {
                    float val = Exparr[j] / sumExp;
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }

            // part c
            for (int i = 0; i < N; i += BLOCK_SIZE)
            {
                for (int j = 0; j < d; j += BLOCK_SIZE)
                {
                    int row_end = std::min(i + BLOCK_SIZE, N);
                    int col_end = std::min(j + BLOCK_SIZE, d);
                    for (int blockRow = i; blockRow < row_end; blockRow++)
                    {
                        for (int blockCol = j; blockCol < col_end; blockCol++)
                        {
                            float val = 0.0;
                            for (int k = 0; k < N; k++)
                            {
                                float qkt = twoDimRead(QK_t, blockRow, k, N);
                                float v = fourDimRead(V, b, h, k, blockCol, H, N, d);
                                val += qkt * v;
                            }
                            fourDimWrite(O, b, h, blockRow, blockCol, H, N, d, val);
                        }
                    }
                }
            }

            // for (int i = 0; i < N; i++)
            // {
            //     for (int j = 0; j < d; j++)
            //     {
            //         float val = 0.0;
            //         for (int k = 0; k < N; k++)
            //         {
            //             float qkt = twoDimRead(QK_t, i, k, N);
            //             float v = fourDimRead(V, b, h, k, j, H, N, d);
            //             val += qkt * v;
            //         }
            //         fourDimWrite(O, b, h, i, j, H, N, d, val);
            //     }
            // }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                               int B, int H, int N, int d)
{

    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Make O Tensor with Shape (B, H, N, d)
    // and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    // Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // Format ORow Tensor into a 1D vector
    //  You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);

// -------- YOUR CODE HERE  -------- //
#pragma omp parallel for collapse(3)
    // We give you a template of the first three loops for your convenience
    // loop over batch
    for (int b = 0; b < B; b++)
    {

        // loop over heads
        for (int h = 0; h < H; h++)
        {
            for (int i = 0; i < N; i++)
            {

                // YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});
                std::vector<float> ORow = formatTensor(ORowTensor);
                // YOUR CODE HERE
                // loop over Embedding Dimensionality (rows of transposed)

                for (int j = 0; j < N; j++)
                {
                    float valQKT = 0.0;
                    // loop over cols
                    for (int k = 0; k < d; k++)
                    {
                        float valQ = fourDimRead(Q, b, h, i, k, H, N, d);
                        float valK = fourDimRead(K, b, h, j, k, H, N, d);
                        valQKT += valQ * valK;
                    }
                    // twoDimWrite(QK_t, i, j, N, valQKT);
                    ORow[j] = valQKT;
                }
                // a row of Qk^t is computed here

                // softmax of a row
                float *Exparr = new float[N];
                float sumExp = 0.0;
                for (int j = 0; j < N; j++)
                {
                    // float val = twoDimRead(QK_t, i, j, N);
                    float val = ORow[j];

                    val = exp(val);
                    Exparr[j] = val;
                    sumExp += val;
                }
                for (int j = 0; j < N; j++)
                {
                    float val = Exparr[j] / sumExp;
                    // twoDimWrite(QK_t, i, j, N, val);
                    ORow[j] = val;
                }
                if (Exparr)
                {
                    delete[] Exparr;
                }
                // softmax of a row done

                // multiplying softmaxed row with V to get attention output
                for (int j = 0; j < d; j++)
                {
                    float val = 0.0;
                    for (int k = 0; k < N; k++)
                    {
                        // float qkt = twoDimRead(QK_t, i, k, N);
                        float qkt = ORow[k];
                        float v = fourDimRead(V, b, h, k, j, H, N, d);
                        val += qkt * v;
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, val);
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //
torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
                               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
                               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
                               torch::Tensor OiTensor, torch::Tensor LTensor, torch::Tensor LiTensor,
                               torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                               int B, int H, int N, int d)
{

    // Create output tensor
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // Initialize according to pseudocode
    std::vector<float> I(N, 0.0f); // li from pseudocode

    for (int b = 0; b < B; ++b)
    {
        for (int h = 0; h < H; ++h)
        {
            std::fill(I.begin(), I.end(), 0.0f);

            // Process K/V blocks (Tc blocks)
            for (int j_start = 0; j_start < N; j_start += Bc)
            {
                int j_end = std::min(j_start + Bc, N);
                int actual_Bc = j_end - j_start;

                // Load K_j block
                std::vector<float> Kj(actual_Bc * d);
                for (int j = 0; j < actual_Bc; ++j)
                {
                    int col_idx = j_start + j;
                    for (int k = 0; k < d; ++k)
                    {
                        Kj[j * d + k] = fourDimRead(K, b, h, col_idx, k, H, N, d);
                    }
                }

                // Load V_j block
                std::vector<float> Vj(actual_Bc * d);
                for (int j = 0; j < actual_Bc; ++j)
                {
                    int col_idx = j_start + j;
                    for (int k = 0; k < d; ++k)
                    {
                        Vj[j * d + k] = fourDimRead(V, b, h, col_idx, k, H, N, d);
                    }
                }

                // Process Q blocks (Tr blocks)
                for (int i_start = 0; i_start < N; i_start += Br)
                {
                    int i_end = std::min(i_start + Br, N);
                    int actual_Br = i_end - i_start;

                    // Load Q_i block
                    std::vector<float> Qi(actual_Br * d);
                    for (int i = 0; i < actual_Br; ++i)
                    {
                        int row_idx = i_start + i;
                        for (int k = 0; k < d; ++k)
                        {
                            Qi[i * d + k] = fourDimRead(Q, b, h, row_idx, k, H, N, d);
                        }
                    }

                    // Compute Sij = Qi * Kj^T
                    std::vector<float> Sij(actual_Br * actual_Bc);
                    for (int i = 0; i < actual_Br; ++i)
                    {
                        for (int j = 0; j < actual_Bc; ++j)
                        {
                            float sum = 0.0f;
                            for (int k = 0; k < d; ++k)
                            {
                                sum += Qi[i * d + k] * Kj[j * d + k];
                            }
                            Sij[i * actual_Bc + j] = sum;
                        }
                    }

                    // Compute Pij = exp(Sij)
                    std::vector<float> Pij(actual_Br * actual_Bc);
                    for (int i = 0; i < actual_Br; ++i)
                    {
                        for (int j = 0; j < actual_Bc; ++j)
                        {
                            Pij[i * actual_Bc + j] = exp(Sij[i * actual_Bc + j]);
                        }
                    }

                    // Compute lij = rowsum(Pij)
                    std::vector<float> lij(actual_Br);
                    for (int i = 0; i < actual_Br; ++i)
                    {
                        float sum = 0.0f;
                        for (int j = 0; j < actual_Bc; ++j)
                        {
                            sum += Pij[i * actual_Bc + j];
                        }
                        lij[i] = sum;
                    }

                    // Update output (strict pseudocode translation)
                    for (int i = 0; i < actual_Br; ++i)
                    {
                        int row_idx = i_start + i;
                        float Inew = I[row_idx] + lij[i];

                        if (Inew > 0)
                        { // Prevent division by zero
                            // Scale existing values
                            for (int k = 0; k < d; ++k)
                            {
                                float current_val = fourDimRead(O, b, h, row_idx, k, H, N, d);
                                current_val *= I[row_idx] / Inew;

                                // Add new contributions
                                for (int j = 0; j < actual_Bc; ++j)
                                {
                                    float p = Pij[i * actual_Bc + j] / Inew;
                                    current_val += p * Vj[j * d + k];
                                }

                                fourDimWrite(O, b, h, row_idx, k, H, N, d, current_val);
                            }
                        }

                        // Update I (li)
                        I[row_idx] = Inew;
                    }
                }
            }
        }
    }
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}
/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
    m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
    m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
    m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
    m.def("twoDimRead", &twoDimRead, "twoDimRead");
    m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
