#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "uthash.h"
#include <chrono>

#include <omp.h>
#include <vector>
#include <string>

// Platform-specific includes for memory mapping
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

typedef unsigned char byte;

#define SIGNATURE_LEN 64

int DENSITY = 21;
int PARTITION_SIZE;

int inverse[256];
const char* alphabet = "CSTPAGNDEQHRKMILVFYW";


void seed_random(char* term, int length);
short random_num(short max);
void Init();

int doc_sig[SIGNATURE_LEN];

int WORDLEN;

typedef struct
{
    char term[100];
    short sig[SIGNATURE_LEN];
    UT_hash_handle hh;
} hash_term;

hash_term* vocab = NULL;

#define MAX_KMERS 8000
short precomputed_sigs[MAX_KMERS][SIGNATURE_LEN];
bool use_precomputed = true;

struct PartitionSignature {
    int doc_id;
    byte signature[SIGNATURE_LEN / 8];
};

struct PartitionWork {
    int seq_idx;
    int offset;
    int length;
    int doc_id;
};

// Structure to hold parsed sequence info
struct SequenceInfo {
    const char* start;  // Pointer into mmap'd memory
    int length;
};

int kmer_to_index(char* term)
{
    int index = 0;
    for (int i = 0; i < WORDLEN; i++)
    {
        index = index * 20 + inverse[(unsigned char)term[i]];
    }
    return index;
}

void index_to_kmer(int index, char* term)
{
    for (int i = WORDLEN - 1; i >= 0; i--)
    {
        term[i] = alphabet[index % 20];
        index /= 20;
    }
    term[WORDLEN] = '\0';
}

short* compute_new_term_sig(char* term, short* term_sig)
{
    seed_random(term, WORDLEN);
    int non_zero = SIGNATURE_LEN * DENSITY / 100;

    int positive = 0;
    while (positive < non_zero / 2)
    {
        short pos = random_num(SIGNATURE_LEN);
        if (term_sig[pos] == 0)
        {
            term_sig[pos] = 1;
            positive++;
        }
    }

    int negative = 0;
    while (negative < non_zero / 2)
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

void precompute_all_signatures()
{
    printf("Pre-computing all %d possible %d-mer signatures...\n", MAX_KMERS, WORDLEN);
    auto start = std::chrono::high_resolution_clock::now();

    char term[10];
    for (int i = 0; i < MAX_KMERS; i++)
    {
        index_to_kmer(i, term);
        memset(precomputed_sigs[i], 0, sizeof(precomputed_sigs[i]));
        compute_new_term_sig(term, precomputed_sigs[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("Pre-computation complete in %f seconds\n", duration.count());
}

short* find_sig_fast(char* term)
{
    int index = kmer_to_index(term);
    return precomputed_sigs[index];
}

short* find_sig_hash(char* term)
{
    hash_term* entry;
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

short* find_sig(char* term)
{
    if (use_precomputed)
        return find_sig_fast(term);
    else
        return find_sig_hash(term);
}

//#define min(a,b) ((a) < (b) ? (a) : (b))

inline int count_partitions(int seq_len)
{
    int count = 0;
    int pos = 0;
    do {
        count++;
        pos += PARTITION_SIZE / 2;
    } while (pos + PARTITION_SIZE / 2 < seq_len);
    return count;
}

int power(int n, int e)
{
    int p = 1;
    for (int j = 0; j < e; j++)
        p *= n;
    return p;
}

// Robust parallel FASTA parser with CRLF handling and safe chunk boundaries
std::vector<SequenceInfo> parse_fasta_parallel(const char* filename, size_t& file_size)
{
    std::vector<SequenceInfo> sequences;

    // --- Memory map the file (platform-specific as before) ---
#ifdef _WIN32
    HANDLE hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return sequences;

    LARGE_INTEGER fileSize;
    GetFileSizeEx(hFile, &fileSize);
    file_size = fileSize.QuadPart;

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) { CloseHandle(hFile); return sequences; }

    const char* mapped = (const char*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!mapped) { CloseHandle(hMapping); CloseHandle(hFile); return sequences; }
#else
    int fd = open(filename, O_RDONLY);
    if (fd == -1) return sequences;

    struct stat sb;
    if (fstat(fd, &sb) == -1) { close(fd); return sequences; }
    file_size = sb.st_size;

    const char* mapped = (const char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) { close(fd); return sequences; }
#endif

    // --- Fallback for very small files ---
    int num_threads = omp_get_max_threads();
    if (file_size < 1024 || num_threads == 1) {
        size_t line_start = 0;
        bool expect_sequence = false;
        for (size_t i = 0; i < file_size; i++) {
            if (mapped[i] == '\n' || mapped[i] == '\r') {
                size_t line_len = i - line_start;
                if (line_len > 0) {
                    if (mapped[line_start] == '>') {
                        expect_sequence = true;
                    }
                    else if (expect_sequence) {
                        sequences.push_back({ mapped + line_start, (int)line_len });
                        expect_sequence = false;
                    }
                }
                if (mapped[i] == '\r' && i + 1 < file_size && mapped[i + 1] == '\n') i++;
                line_start = i + 1;
            }
        }
        return sequences;
    }

    // --- Parallel parsing for larger files ---
    size_t chunk_size = (file_size + num_threads - 1) / num_threads;
    std::vector<std::vector<SequenceInfo>> thread_results(num_threads);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t start = tid * chunk_size;
        size_t end = std::min<size_t>(file_size, start + chunk_size);

        // Extend end to next newline so we don’t cut a line in half
        while (end < file_size && mapped[end] != '\n' && mapped[end] != '\r') end++;

        // Adjust start to line boundary (except for thread 0)
        if (tid > 0) {
            while (start < end && mapped[start] != '\n' && mapped[start] != '\r') start++;
            if (start < end) start++;
        }

        bool expect_sequence = false;
        size_t line_start = start;
        auto& my_results = thread_results[tid];

        for (size_t i = start; i < end; i++) {
            if (mapped[i] == '\n' || mapped[i] == '\r') {
                size_t line_len = i - line_start;
                if (line_len > 0) {
                    if (mapped[line_start] == '>') {
                        expect_sequence = true;
                    }
                    else if (expect_sequence) {
                        // Trim trailing CR if present
                        while (line_len > 0 && (mapped[line_start + line_len - 1] == '\r' ||
                            mapped[line_start + line_len - 1] == '\n'))
                            line_len--;

                        if (line_len > 0) {
                            my_results.push_back({ mapped + line_start, (int)line_len });
                        }
                        expect_sequence = false;
                    }
                }
                if (mapped[i] == '\r' && i + 1 < end && mapped[i + 1] == '\n') i++;
                line_start = i + 1;
            }
        }

        // *** NEW: handle dangling header at chunk end ***
        if (expect_sequence && line_start < end) {
            size_t line_len = end - line_start;
            while (line_len > 0 && (mapped[line_start + line_len - 1] == '\r' ||
                mapped[line_start + line_len - 1] == '\n'))
                line_len--;
            if (line_len > 0) {
                my_results.push_back({ mapped + line_start, (int)line_len });
            }
        }

        // If ends
        if (expect_sequence) {
            size_t line_len = end - line_start;
            while (line_len > 0 && (mapped[line_start + line_len - 1] == '\r' ||
                mapped[line_start + line_len - 1] == '\n'))
                line_len--;

            if (line_len > 0) {
                my_results.push_back({ mapped + line_start, (int)line_len });
            }
            expect_sequence = false;
        }

    }

    // After all threads finish, merge results as before


    // Merge thread results
    size_t total = 0;
    for (auto& v : thread_results) total += v.size();
    sequences.reserve(total);
    for (auto& v : thread_results) {
        sequences.insert(sequences.end(), v.begin(), v.end());
    }

   


    return sequences;
}


// Parse FASTA file in parallel using memory mapping
/*
std::vector<SequenceInfo> parse_fasta_parallel(const char* filename, size_t& file_size)
{
    std::vector<SequenceInfo> sequences;

#ifdef _WIN32
    // Windows memory mapping
    HANDLE hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Error: failed to open file %s\n", filename);
        return sequences;
    }

    LARGE_INTEGER fileSize;
    GetFileSizeEx(hFile, &fileSize);
    file_size = fileSize.QuadPart;

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        CloseHandle(hFile);
        fprintf(stderr, "Error: failed to create file mapping\n");
        return sequences;
    }

    const char* mapped = (const char*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!mapped) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
        fprintf(stderr, "Error: failed to map file\n");
        return sequences;
    }
#else
    // Unix memory mapping
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "Error: failed to open file %s\n", filename);
        return sequences;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        fprintf(stderr, "Error: failed to get file size\n");
        return sequences;
    }
    file_size = sb.st_size;

    const char* mapped = (const char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        fprintf(stderr, "Error: failed to map file\n");
        return sequences;
    }
#endif

    printf("File mapped: %zu bytes\n", file_size);

    // PARALLEL PARSING: Divide file into chunks for parallel scanning
    int num_threads = omp_get_max_threads();
    size_t chunk_size = file_size / num_threads;

    std::vector<std::vector<size_t>> thread_seq_positions(num_threads);

    printf("Scanning for sequences in parallel...\n");

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t start = tid * chunk_size;
        size_t end = (tid == num_threads - 1) ? file_size : (tid + 1) * chunk_size;

        // Adjust start to beginning of a line (except for thread 0)
        if (tid > 0) {
            while (start < end && mapped[start] != '\n') start++;
            if (start < end) start++; // Move past the newline
        }

        std::vector<size_t>& my_positions = thread_seq_positions[tid];

        // Scan for sequence lines
        bool expect_sequence = false;
        size_t line_start = start;

        for (size_t i = start; i < end; i++) {
            if (mapped[i] == '\n') {
                if (expect_sequence && i > line_start) {
                    my_positions.push_back(line_start);
                    my_positions.push_back(i - line_start);
                    expect_sequence = false;
                }
                else if (i > line_start && mapped[line_start] == '>') {
                    expect_sequence = true;
                }
                line_start = i + 1;
            }
        }

        // Handle last line
        if (expect_sequence && line_start < end && tid == num_threads - 1) {
            my_positions.push_back(line_start);
            my_positions.push_back(end - line_start);
        }
    }

    // Count total sequences
    size_t total_sequences = 0;
    for (const auto& positions : thread_seq_positions) {
        total_sequences += positions.size() / 2;
    }

    printf("Found %zu sequences\n", total_sequences);
    sequences.reserve(total_sequences);

    // Merge results
    for (const auto& positions : thread_seq_positions) {
        for (size_t i = 0; i < positions.size(); i += 2) {
            SequenceInfo info;
            info.start = mapped + positions[i];
            info.length = (int)positions[i + 1];
            sequences.push_back(info);
        }
    }

    return sequences;
}
*/

int main(int argc, char* argv[])
{
    const char* filename;

    if (argc > 1) {
        filename = argv[1];
    }
    else {
        filename = "test.fasta";
    }

    WORDLEN = 3;
    PARTITION_SIZE = 16;
    int WORDS = power(20, WORDLEN);

    for (int i = 0; i < strlen(alphabet); i++)
        inverse[alphabet[i]] = i;

    if (use_precomputed)
    {
        precompute_all_signatures();
    }

    // START TIMING HERE
    auto start = std::chrono::high_resolution_clock::now();

    // ========== PARALLEL FILE READING ==========
    size_t file_size;
    std::vector<SequenceInfo> sequences = parse_fasta_parallel(filename, file_size);

    if (sequences.empty()) {
        fprintf(stderr, "No sequences found!\n");
        return 1;
    }

    printf("Processing %zu sequences...\n", sequences.size());

    // ========== BUILD WORK QUEUE IN PARALLEL ==========
    printf("Building work queue in parallel...\n");

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<PartitionWork>> thread_work(num_threads);
    std::vector<int> thread_doc_counts(num_threads, 0);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::vector<PartitionWork>& my_work = thread_work[tid];

        int sequences_per_thread = (sequences.size() + num_threads - 1) / num_threads;
        my_work.reserve(sequences_per_thread * 10);

#pragma omp for schedule(static) nowait
        for (int seq_idx = 0; seq_idx < (int)sequences.size(); seq_idx++) {
            int seq_len = sequences[seq_idx].length;
            int offset = 0;

            do {
                my_work.push_back({
                    seq_idx,
                    offset,
                    std::min(PARTITION_SIZE, seq_len - offset),
                    -1
                    });
                offset += PARTITION_SIZE / 2;
            } while (offset + PARTITION_SIZE / 2 < seq_len);
        }

        thread_doc_counts[tid] = (int)my_work.size();
    }

    // Calculate doc_id offsets
    std::vector<int> doc_id_offsets(num_threads);
    doc_id_offsets[0] = 0;
    for (int i = 1; i < num_threads; i++) {
        doc_id_offsets[i] = doc_id_offsets[i - 1] + thread_doc_counts[i - 1];
    }
    int total_partitions = doc_id_offsets[num_threads - 1] + thread_doc_counts[num_threads - 1];

    printf("Total partitions: %d\n", total_partitions);

    // Merge work queues in parallel
    std::vector<PartitionWork> all_work(total_partitions);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int offset = doc_id_offsets[tid];

        for (size_t i = 0; i < thread_work[tid].size(); i++) {
            all_work[offset + i] = thread_work[tid][i];
            all_work[offset + i].doc_id = offset + i;
        }
    }

    // ========== PARALLEL SIGNATURE COMPUTATION ==========
    printf("Processing %zu partitions in parallel...\n", all_work.size());

    std::vector<PartitionSignature> results(all_work.size());

#pragma omp parallel for schedule(static)
    for (int work_idx = 0; work_idx < (int)all_work.size(); work_idx++)
    {
        const PartitionWork& work = all_work[work_idx];
        const char* seq_start = sequences[work.seq_idx].start + work.offset;

        PartitionSignature result;
        result.doc_id = work.doc_id;

        int doc_sig[SIGNATURE_LEN];
        memset(doc_sig, 0, sizeof(doc_sig));

        for (int i = 0; i < work.length - WORDLEN + 1; i++)
        {
            const char* term = seq_start + i;

            int index = inverse[(unsigned char)term[0]] * 400 +
                inverse[(unsigned char)term[1]] * 20 +
                inverse[(unsigned char)term[2]];

            short* term_sig = precomputed_sigs[index];
            for (int j = 0; j < SIGNATURE_LEN; j++)
            {
                doc_sig[j] += term_sig[j];
            }
        }

        for (int i = 0; i < SIGNATURE_LEN; i += 8)
        {
            byte c = 0;
            for (int j = 0; j < 8; j++)
                c |= (doc_sig[i + j] > 0) << (7 - j);
            result.signature[i / 8] = c;
        }

        results[work_idx] = result;
    }

    // ========== PARALLEL OUTPUT BUFFER CONSTRUCTION ==========
    printf("Building output buffer in parallel...\n");

    const size_t record_size = sizeof(int) + SIGNATURE_LEN / 8;  // doc_id + signature
    const size_t output_size = total_partitions * record_size;

    // Allocate output buffer
    std::vector<byte> output_buffer(output_size);

    // Each thread writes its portion of results to the buffer in parallel
#pragma omp parallel for schedule(static)
    for (int i = 0; i < total_partitions; i++)
    {
        byte* dest = output_buffer.data() + (i * record_size);

        // Write doc_id
        memcpy(dest, &results[i].doc_id, sizeof(int));
        dest += sizeof(int);

        // Write signature
        memcpy(dest, results[i].signature, SIGNATURE_LEN / 8);
    }

    // ========== SINGLE SEQUENTIAL WRITE ==========
    printf("Writing output file...\n");

    char outfile[256];
    sprintf_s(outfile, 256, "%s.part%d_sigs%02d_%d", filename, PARTITION_SIZE, WORDLEN, SIGNATURE_LEN);

    FILE* sig_file;
    fopen_s(&sig_file, outfile, "wb");

    // Single large write - much faster than many small writes!
    fwrite(output_buffer.data(), 1, output_size, sig_file);

    fclose(sig_file);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    printf("%s %f seconds\n", filename, duration.count());
    printf("Method: Fully parallel (mmap I/O, parallel output buffer)\n");
    printf("Threads used: %d\n", omp_get_max_threads());
    printf("Output size: %.2f MB\n", output_size / (1024.0 * 1024.0));

    return 0;
}