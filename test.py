from src.cfg import constants

gene_id = "xXxeukvoKOwyhl8i8A54SGLTaJQ"

if __name__ == "__main__":
    results_path = f'{constants.SOTA_ROOT}/results/{gene_id}_results.txt'
    print(results_path)
    with open(results_path, 'r') as file:
        results = file.read()
    results = results.split(',')
    fitness = [float(r.strip()) for r in results]

    print(results)
    print(fitness)