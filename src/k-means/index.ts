import KMeans from "./k-means";
import KMeansAutoSolver from "./k-means-auto-solver";
import example_data from "./data";
console.log("===============================================\n");
console.log("Testing centroid generation:");
console.log("===============================================\n");
const ex_randomCentroids_solver = new KMeans(
  2,
  example_data.example_randomCentroids
);
console.log("Randomly initialized centroids: ");
console.log(ex_randomCentroids_solver.centroids);
console.log("\n-----------------------------------------------\n\n");

console.log("Solving example: 2d data with 3 clusters:");
console.log("===============================================\n");
const ex_1_solver = new KMeans(3, example_data.example_2d3k);
const ex_1_centroids = ex_1_solver.solve();
console.log(ex_1_centroids);
console.log("\n-----------------------------------------------\n\n");

console.log("Iteration log for 2d data with 3 clusters:");
console.log("------------------------------------------");
ex_1_solver.iterationLogs.forEach(log => console.log(log));

console.log("Test 2d data with 3 clusters five times:");
console.log("----------------------------------------");
for (let i = 0; i < 5; i++) {
  ex_1_solver.reset();
  const solution = ex_1_solver.solve();
  console.log(solution);
}

console.log("Solution for 3d data with 3 clusters:");
console.log("===============================================\n");
const ex_2_solver = new KMeans(3, example_data.example_3d3k);
const ex_2_centroids = ex_2_solver.solve();
console.log(ex_2_centroids);

console.log("Solving example: 2d data with unknown clusters:");
console.log("===============================================\n");

const ex_3_solver = new KMeansAutoSolver(1, 5, 5, example_data.example_2dnk);
const ex_3_solution = ex_3_solver.solve();
console.log(ex_3_solution);
