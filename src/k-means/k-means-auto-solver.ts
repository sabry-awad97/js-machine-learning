import KMeans, { Result } from "./k-means";

interface Solution extends Result {
  k: number;
  currentTrial: number;
}

class KMeansAutoSolver {
  best: Solution | null = null;
  log!: Solution[];

  constructor(
    private kMin = 1,
    private kMax = 5,
    private maxTrials = 5,
    private data: number[][]
  ) {
    this.reset();
  }

  reset() {
    this.best = null;
    this.log = [];
  }

  solve(maxIterations = 1000) {
    for (let k = this.kMin; k < this.kMax; k++) {
      for (
        let currentTrial = 0;
        currentTrial < this.maxTrials;
        currentTrial++
      ) {
        const solver = new KMeans(k, this.data);
        // Add k and currentTrial number to the solution before logging;
        const solution = { ...solver.solve(maxIterations), k, currentTrial };
        this.log.push(solution);
        if (this.best === null || solution.error! < this.best.error!) {
          this.best = solution;
        }
      }
    }

    return this.best;
  }
}
export default KMeansAutoSolver;
