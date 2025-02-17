CS 179
Assignment 2

Due: Wednesday, April 17, 2019 - 3:00 PM.

Put all answers in a file called README.txt. After answering all of the
questions, list how long part 1 and part 2 took. Feel free to leave any other
feedback.

========================================
NOTE: New submission method!

Instead of emailing us the solution, put a zip file in your home directory
on Titan, in the format:

lab2_2019_submission.zip


Your submission should be a single archive file (.zip)
with your README file and all code.

========================================


PART 1

Question 1.1: Latency Hiding (5 points)
---------------------------------------

Approximately how many arithmetic instructions does it take to hide the latency
of a single arithmetic instruction on a GK110?

Assume all of the arithmetic instructions are independent (ie have no
instruction dependencies).

You do not need to consider the number of execution cores on the chip.

Hint: What is the latency of an arithmetic instruction? How many instructions
can a GK110 begin issuing in 1 clock cycle (assuming no dependencies)?

%%
[https://github.com/Dama624/Caltech-CS179/tree/master/Problem%20Set%202]
The latency of an arithmetic instruction is 10 ns. The number of instructions per cycle for a GK110 is 8
(4 warps each clock, up to 2 instructions in each warp). A GPU clock is 1 GHz (1 clock per ns).
Thus, for 1 ns, the GPU undergoes 1 cycle, and 8 instructions in this cycle.

Since an arithmetic instruction takes 10 ns, 10 * 8 = 80 instructions can take place in the span of an arithmetic instruction.
Thus, it takes 80 instructions to hide the latency of a single arithmetic instruction.


Question 1.2: Thread Divergence (6 points)
------------------------------------------

Let the block shape be (32, 32, 1).

(a)
int idx = threadIdx.y + blockSize.y * threadIdx.x;
if (idx % 32 < 16)
    foo();
else
    bar();

Does this code diverge? Why or why not?

%%
No, idx % 32 = (threadIdx.y + 32 * threadIdx.x) % 32 = threadIdx.y

(b)
const float pi = 3.14;
float result = 1.0;
for (int i = 0; i < threadIdx.x; i++)
    result *= pi;

Does this code diverge? Why or why not? (This is a bit of a trick question,
either "yes" or "no can be a correct answer with appropriate explanation.)

%%
Yes, the for loop range is different.
[https://devtalk.nvidia.com/default/topic/1031723/warp-divergence-triggered-by-for-loop/]

Question 1.3: Coalesced Memory Access (9 points)
------------------------------------------------

Let the block shape be (32, 32, 1). Let data be a (float *) pointing to global
memory and let data be 128 byte aligned (so data % 128 == 0).

Consider each of the following access patterns.

(a)
data[threadIdx.x + blockSize.x * threadIdx.y] = 1.0;

Is this write coalesced? How many 128 byte cache lines does this write to?

%%
Yes. For a single warp, this writes to one 128-byte cache line.

(b)
data[threadIdx.y + blockSize.y * threadIdx.x] = 1.0;

Is this write coalesced? How many 128 byte cache lines does this write to?

%%
No, address is (32 * 0, 32 * 1, 32 * 2, ... 32 * 31) in first warp, 32 128 byte cache lines writing are needed per warp.

(c)
data[1 + threadIdx.x + blockSize.x * threadIdx.y] = 1.0;

Is this write coalesced? How many 128 byte cache lines does this write to?

%%
No, two 128 byte cache lines writing needed per warp.

Question 1.4: Bank Conflicts and Instruction Dependencies (15 points)
---------------------------------------------------------------------

Let's consider multiplying a 32 x 128 matrix with a 128 x 32 element matrix.
This outputs a 32 x 32 matrix. We'll use 32 ** 2 = 1024 threads and each thread
will compute 1 output element. Although its not optimal, for the sake of
simplicity let's use a single block, so grid shape = (1, 1, 1),
block shape = (32, 32, 1).

For the sake of this problem, let's assume both the left and right matrices have
already been stored in shared memory are in column major format. This means the
element in the ith row and jth column is accessible at lhs[i + 32 * j] for the
left hand side and rhs[i + 128 * j] for the right hand side.

This kernel will write to a variable called output stored in shared memory.

Consider the following kernel code:

int i = threadIdx.x;
int j = threadIdx.y;
for (int k = 0; k < 128; k += 2) {
    output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
    output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
}

(a)
Are there bank conflicts in this code? If so, how many ways is the bank conflict
(2-way, 4-way, etc)?

%%
No, each warp compute one column in output matrix, take `column major` in mind.

(b)
Expand the inner part of the loop (below)

output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];

into "psuedo-assembly" as was done in the coordinate addition example in lecture
4.

There's no need to expand the indexing math, only to expand the loads, stores,
and math. Notably, the operation a += b * c can be computed by a single
instruction called a fused multiply add (FMA), so this can be a single
instruction in your "psuedo-assembly".

Hint: Each line should expand to 5 instructions.

%%
lhs0 = lhs[i + 32 * k];
rhs0 = rhs[k + 128 * j];
lhs1 = lhs[i + 32 * (k + 1)];
rhs1 = rhs[(k + 1) + 128 * j];
O = output[i + 32 * j];
FMA on lhs0, rhs0, O;
FMA on lhs1, rhs1, O;
Write O to output[i + 32 * j];

(c)
Identify pairs of dependent instructions in your answer to part b.

%%
two FMA op is dependent

(d)
Rewrite the code given at the beginning of this problem to minimize instruction
dependencies. You can add or delete instructions (deleting an instruction is a
valid way to get rid of a dependency!) but each iteration of the loop must still
process 2 values of k.

%%
intuition: use register to boost speed

int i = threadIdx.x;
int j = threadIdx.y;
int temp = 0;
for (int k = 0; k < 128; k += 2) {
    temp += lhs[i + 32 * k] * rhs[k + 128 * j];
    temp += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
}
output[i + 32 * j] = temp

(e)
Can you think of any other anything else you can do that might make this code
run faster?




PART 2 - Matrix transpose optimization (65 points)
--------------------------------------------------

Optimize the CUDA matrix transpose implementations in transpose_cuda.cu. Read
ALL of the TODO comments. Matrix transpose is a common exercise in GPU
optimization, so do not search for existing GPU matrix transpose code on the
internet.

Your transpose code only need to be able to transpose square matrices where the
side length is a multiple of 64.

The initial implementation has each block of 1024 threads handle a 64x64 block
of the matrix, but you can change anything about the kernel if it helps obtain
better performance.

The main method of transpose.cc already checks for correctness for all transpose
results, so there should be an assertion failure if your kernel produces incorrect
output.

The purpose of the shmemTransposeKernel is to demonstrate proper usage of global
and shared memory. The optimalTransposeKernel should be built on top of
shmemTransposeKernel and should incorporate any "tricks" such as ILP, loop
unrolling, vectorized IO, etc that have been discussed in class.

You can compile and run the code by running

make transpose
./transpose

and the build process was tested on minuteman. If this does not work on haru for
you, be sure to add the lines

export PATH=/usr/local/cuda-6.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:$LD_LIBRARY_PATH

to your ~/.profile file (and then exit and ssh back in to restart your shell).

On OS X, you may have to run or add to your .bash_profile the command

export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/local/cuda/lib/

in order to get dynamic library linkage to work correctly.

The transpose program takes 2 optional arguments: input size and method. Input
size must be one of -1, 512, 1024, 2048, 4096, and method must be one all,
cpu, gpu_memcpy, naive, shmem, optimal. Input size is the first argument and
defaults to -1. Method is the second argument and defaults to all. You can pass
input size without passing method, but you cannot pass method without passing an
input size.

Examples:
./transpose
./transpose 512
./transpose 4096 naive
./transpose -1 optimal

Copy paste the output of ./transpose.cc into README.txt once you are done.
Describe the strategies used for performance in either block comments over the
kernel (as done for naiveTransposeKernel) or in README.txt.




BONUS (+5 points, maximum set score is 100 even with bonus)
--------------------------------------------------------------------------------

Mathematical scripting environments such as Matlab or Python + Numpy often
encourage expressing algorithms in terms of vector operations because they offer
a convenient and performant interface. For instance, one can add 2 n-component
vectors (a and b) in Numpy with c = a + b.

This is often implemented with something like the following code:

void vec_add(float *left, float *right, float *out, int size) {
    for (int i = 0; i < size; i++)
        out[i] = left[i] + right[i];
}

Consider the code

a = x + y + z

where x, y, z are n-component vectors.

One way this could be computed would be

vec_add(x, y, a, n);
vec_add(a, z, a, n);

In what ways is this code (2 calls to vec_add) worse than the following?

for (int i = 0; i < n; i++)
    a[i] = x[i] + y[i] + z[i];

List at least 2 ways (you don't need more than a sentence or two for each way).
