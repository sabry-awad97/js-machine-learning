import { KNN } from "./knn";
import { weight_height } from "./data";

console.log("Testing height and weight with k=5");
console.log("==========================");

const data = weight_height.map(({ point }) => point);
const labels = weight_height.map(({ label }) => label);

const solver1 = new KNN(5, data, labels);
console.log("Testing a 'definitely male' point:");
console.log(solver1.predict([200, 75]));
console.log("\nTesting a 'probably male' point:");
console.log(solver1.predict([170, 70]));
console.log("\nTesting a 'totally uncertain' point:");
console.log(solver1.predict([140, 64]));
console.log("\nTesting a 'probably female' point:");
console.log(solver1.predict([130, 63]));
console.log("\nTesting a 'definitely female' point:");
console.log(solver1.predict([120, 60]));
