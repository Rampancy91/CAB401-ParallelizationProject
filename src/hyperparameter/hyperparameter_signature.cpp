#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "uthash.h"
#include <chrono>

#include <omp.h> //OpenMP
#include <vector> //Storing multiple results


typedef unsigned char byte;

#define SIGNATURE_LEN 64

int DENSITY  = 21;
int PARTITION_SIZE;

int inverse[256];
const char* alphabet = "CSTPAGNDEQHRKMILVFYW";


void seed_random(char* term, int length);
short random_num(short max);
void Init();

int doc_sig[SIGNATURE_LEN];

int WORDLEN;
FILE *sig_file;

typedef struct
{
    char term[100];
    short sig[SIGNATURE_LEN];
    UT_hash_handle hh;
} hash_term;

hash_term *vocab = NULL;

//pre-compute signatures array
// Only 20^3 = 8000 possible combinations

#define MAX_KMERS 8000
#define MAX_KMERS 160000      // NEW - supports up to k=4
#define MAX_KMERS 3200000  // Support up to k=5

short precomputed_sigs[MAX_KMERS][SIGNATURE_LEN];
bool use_precomputed = true; //To toggle between methods

//Structure to hold partitions signature data
struct PartitionSignature {
    int doc_id;
    byte signature[SIGNATURE_LEN / 8];
};

//K-mer index functions
//Convert kmer to unique index
int kmer_to_index(char* term)
{
    int index = 0;
    for (int i = 0; i < WORDLEN; i++)
    {
        index = index * 20 + inverse[(unsigned char)term[i]];
    }
    return index;
}

//Convert index back to k-mer
void index_to_kmer(int index, char* term)
{
    for (int i = WORDLEN - 1; i >= 0; i--)
    {
        term[i] = alphabet[index % 20];
        index /= 20;
    }
    term[WORDLEN] = '\0';
}

short* compute_new_term_sig(char* term, short *term_sig)
{
    seed_random(term, WORDLEN);

    int non_zero = SIGNATURE_LEN * DENSITY/100;

    int positive = 0;
    while (positive < non_zero/2)
    {
        short pos = random_num(SIGNATURE_LEN);
        if (term_sig[pos] == 0) 
	{
            term_sig[pos] = 1;
            positive++;
        }
    }

    int negative = 0;
    while (negative < non_zero/2)
    {
        short pos = random_num(SIGNATURE_LEN);
        if (term_sig[pos] == 0) 
	{
            term_sig[pos] = -1;
            negative++;
        }
    }
    return term_sig;
}

//Pre compute all 8000 possible signatures //Opted to parallelise this in order to enable reasonable testing of larger k-mer sizes
void precompute_all_signatures()
{
    printf("Pre-computing all %d possible %d-mer signatures...\n", MAX_KMERS, WORDLEN);

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < MAX_KMERS; i++)
    {
        char term[10];  
        index_to_kmer(i, term);
        memset(precomputed_sigs[i], 0, sizeof(precomputed_sigs[i]));
        compute_new_term_sig(term, precomputed_sigs[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    printf("Pre-computation complete in %f seconds\n", duration.count());
}

// New version -  signature lookup using pre-computed array
short* find_sig_fast(char* term)
{
    int index = kmer_to_index(term);
    return precomputed_sigs[index];
}

// Old version - hash table look up
short *find_sig_hash(char* term)
{
    hash_term *entry;
    HASH_FIND(hh, vocab, term, WORDLEN, entry);
    if (entry == NULL)
    {
        entry = (hash_term*)malloc(sizeof(hash_term));
        strncpy_s(entry->term, sizeof(entry->term), term, WORDLEN);
        memset(entry->sig, 0, sizeof(entry->sig));
        compute_new_term_sig(term, entry->sig);
        HASH_ADD(hh, vocab, term, WORDLEN, entry);
    }

    return entry->sig;
}

//Use selected method
short* find_sig(char* term)
{
    if (use_precomputed)
        return find_sig_fast(term);
    else
        return find_sig_hash(term);
}

// MODIFIED: pass doc_sig as parameter to make thread safe

void signature_add(char* term, int* doc_sig)  
{
    short* term_sig = find_sig(term);
    for (int i = 0; i < SIGNATURE_LEN; i++)
        doc_sig[i] += term_sig[i];
}

//int doc = 0; //unused for now

PartitionSignature compute_signature(char* sequence, int length, int doc_id)
{
    PartitionSignature result;
    result.doc_id = doc_id;

    int doc_sig[SIGNATURE_LEN];
    memset(doc_sig, 0, sizeof(doc_sig));

    // OPTIMIZED: Inline the kmer lookup and accumulation
    for (int i = 0; i < length - WORDLEN + 1; i++)
    {
        // Calculate kmer index directly
        char* term = sequence + i;
        int index = 0;
        for (int k = 0; k < WORDLEN; k++)
        {
            index = index * 20 + inverse[(unsigned char)term[k]];
        }

        // Get signature and accumulate directly
        short* term_sig = precomputed_sigs[index];
        for (int j = 0; j < SIGNATURE_LEN; j++)
        {
            doc_sig[j] += term_sig[j];
        }
    }

    // Flatten signature to bytes
    for (int i = 0; i < SIGNATURE_LEN; i += 8)
    {
        byte c = 0;
        for (int j = 0; j < 8; j++)
            c |= (doc_sig[i + j] > 0) << (7 - j);
        result.signature[i / 8] = c;
    }

    return result;
}

#define min(a,b) ((a) < (b) ? (a) : (b))

// MODIFIED: Return vector of signatures instead of writing to file, can't have multiple threads writing to file
std::vector<PartitionSignature> partition(char* sequence, int length, int starting_doc_id)
{
    std::vector<PartitionSignature> signatures;
    int i = 0;
    int doc_offset = 0;

    do
    {
        PartitionSignature sig = compute_signature(
            sequence + i,
            min(PARTITION_SIZE, length - i),
            starting_doc_id + doc_offset //Calculate correct doc_ID
        );
        signatures.push_back(sig);
        doc_offset++;
        i += PARTITION_SIZE / 2;
    } while (i + PARTITION_SIZE / 2 < length);

    return signatures;
}

int power(int n, int e)
{
    int p = 1;
    for (int j=0; j<e; j++)
        p *= n;
    return p;
}

int main(int argc, char* argv[])
{
    const char* filename = "qut2.fasta"; 

    if (argc > 1) {
        filename = argv[1];
    }

    printf("Hyperparameter Tuning Experiment\n");
    printf("=================================\n");
    printf("File: %s\n", filename);
    printf("Threads: %d\n\n", omp_get_max_threads());

    // Open CSV file for results
    FILE* csv;
    fopen_s(&csv, "tuning_results.csv", "w");
    fprintf(csv, "k_mer,partition_size,time_seconds,memory_MB\n");

    // Test configurations
    int k_values[] = { 2, 3, 4 , 5};
    int p_values[] = { 16, 32, 64, 128, 256 };

    for (int k : k_values)
    {
        for (int p : p_values)
        {
            printf("Testing k=%d, partition=%d... ", k, p);
            fflush(stdout);

            // Set parameters
            WORDLEN = k;
            PARTITION_SIZE = p;
            int max_kmers = (int)pow(20, WORDLEN);

            // Setup inverse mapping
            for (int i = 0; i < strlen(alphabet); i++)
                inverse[alphabet[i]] = i;

            // Pre-compute signatures (don't time this - it's one-time setup)
            if (use_precomputed)
            {
                // Reallocate precomputed_sigs if needed for different k
                // For simplicity, assume global array is large enough
                // or allocate dynamically here

                char term[10];
                for (int i = 0; i < max_kmers; i++)
                {
                    index_to_kmer(i, term);
                    memset(precomputed_sigs[i], 0, sizeof(precomputed_sigs[i]));
                    compute_new_term_sig(term, precomputed_sigs[i]);
                }
            }

            // START TIMING
            auto start = std::chrono::high_resolution_clock::now();

            // Read file
            FILE* file;
            errno_t OK = fopen_s(&file, filename, "r");
            if (OK != 0)
            {
                fprintf(stderr, "Error: failed to open file %s\n", filename);
                return 1;
            }

            std::vector<std::string> sequences;
            char buffer[10000];

            while (!feof(file))
            {
                if (fgets(buffer, 10000, file) == NULL) break;
                if (fgets(buffer, 10000, file) == NULL) break;
                int n = (int)strlen(buffer) - 1;
                if (n > 0) {
                    buffer[n] = 0;
                    sequences.push_back(std::string(buffer));
                }
            }
            fclose(file);

            // Pre-calculate doc IDs
            std::vector<int> doc_id_starts(sequences.size());
            doc_id_starts[0] = 0;

            for (size_t i = 1; i < sequences.size(); i++) {
                int seq_len = (int)sequences[i - 1].length();
                int num_partitions = 0;
                int pos = 0;
                do {
                    num_partitions++;
                    pos += PARTITION_SIZE / 2;
                } while (pos + PARTITION_SIZE / 2 < seq_len);

                doc_id_starts[i] = doc_id_starts[i - 1] + num_partitions;
            }

            // Process sequences in parallel
            std::vector<std::vector<PartitionSignature>> all_signatures(sequences.size());

#pragma omp parallel for schedule(guided, 64)
            for (int seq_idx = 0; seq_idx < (int)sequences.size(); seq_idx++)
            {
                int starting_doc = doc_id_starts[seq_idx];
                char* seq = (char*)sequences[seq_idx].c_str();
                all_signatures[seq_idx] = partition(seq, sequences[seq_idx].length(), starting_doc);
            }

            // Write results to file
            char outfile[256];
            sprintf_s(outfile, 256, "temp_k%d_p%d.sig", k, p);
            FILE* sig_file;
            fopen_s(&sig_file, outfile, "wb");

            for (size_t seq_idx = 0; seq_idx < sequences.size(); seq_idx++)
            {
                for (const auto& sig : all_signatures[seq_idx])
                {
                    fwrite(&sig.doc_id, sizeof(int), 1, sig_file);
                    fwrite(sig.signature, sizeof(byte), SIGNATURE_LEN / 8, sig_file);
                }
            }
            fclose(sig_file);

            // END TIMING
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            double memory_mb = (max_kmers * SIGNATURE_LEN * sizeof(short)) / (1024.0 * 1024.0);

            printf("%.3fs (%.1f MB)\n", duration.count(), memory_mb);

            // Write to CSV
            fprintf(csv, "%d,%d,%.6f,%.2f\n", k, p, duration.count(), memory_mb);
            fflush(csv);

            // Clean up temp file
            remove(outfile);
        }
        printf("\n");
    }

    fclose(csv);

    printf("\nExperiment complete! Results saved to tuning_results.csv\n");
    printf("Open in Excel or use Python/R to analyze\n");

    return 0;
}