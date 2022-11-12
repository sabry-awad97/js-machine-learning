type Point = number[];

export interface Result {
  centroids: number[][];
  iteration: number;
  error: number | null;
  didReachSteadyState: boolean;
}

/**
 * Calculate the mean of an array of numbers.
 */
const mean = (numbers: number[]) =>
  numbers.reduce((sum, val) => sum + val, 0) / numbers.length;

/**
 * Calculate the distance between two points.
 */
const distance = (a: Point, b: Point): number =>
  Math.sqrt(
    a
      .map((aPoint, i) => b[i] - aPoint)
      .reduce((sumOfSquares, diff) => sumOfSquares + diff * diff, 0)
  );

export default class KMeans {
  /**
   * error, which records the root mean square error (RMSE) of the points' distance
   * to their centroids for the current iteration
   */
  private error: number | null = null;

  /**
   * iterations, a simple counter of how many iterations the solver has run, starting from 0 .
   */
  private iterations = 0;

  iterationLogs!: Result[];

  /**
   * centroids, which will store the solver's candidates for the
   * k centroids at the current iteration .
   */
  centroids!: number[][];

  /**
   * centroidAssignments, which is an array of data point index numbers
   * that map to a centroid index number .
   * the index is the index of the point,
   * and the value is the index of the centroid
   */
  private centroidAssignments!: number[];
  /**
   * @param k the desired number of clusters as the algorithm's output.
   * @param data an array of data points that the algorithm will process.
   */
  constructor(private k: number, private data: Point[]) {
    this.reset();
  }
  /**
   * Resets the solver state; use this if you wish to run the
   * same solver instance again with the same data points
   * but different initial conditions.
   */
  reset() {
    this.error = null;
    this.iterations = 0;
    this.iterationLogs = [];
    this.centroids = this.initRandomCentroids();
    this.centroidAssignments = [];
  }

  /**
   * Determines the number of dimensions in the dataset.
   */
  getDimensionality(): number {
    const [point] = this.data;
    return point.length;
  }

  /**
   * For a given dimension in the dataset, determine the minimum
   * and maximum value. This is used during random initialization
   * to make sure the random centroids are in the same range as
   * the data.
   */
  getRangeForDimension(n: number) {
    const values = this.data.map(point => point[n]);
    return {
      min: Math.min.apply(null, values),
      max: Math.max.apply(null, values),
    };
  }

  /**
   * Get ranges for all dimensions.
   * @see getRangeForDimension
   * @returns Array whose indices are the dimension number and whose members
   * are the output of getRangeForDimension
   */
  getAllDimensionRanges() {
    const dimensionRanges = [];
    const dimensionality = this.getDimensionality();
    for (let dimension = 0; dimension < dimensionality; dimension++) {
      dimensionRanges[dimension] = this.getRangeForDimension(dimension);
    }
    return dimensionRanges;
  }

  /**
   * Initializes random centroids, using the ranges of the data
   * to set minimum and maximum bounds for the centroids.
   * You may inspect the output of this method if you need to debug
   * random initialization, otherwise this is an internal method.
   * @see getAllDimensionRanges
   * @see getRangeForDimension
   */
  initRandomCentroids() {
    const dimensionality = this.getDimensionality();
    const dimensionRanges = this.getAllDimensionRanges();
    const centroids = [];

    for (let i = 0; i < this.k; i++) {
      // Since each dimension has its own range, create a placeholder at first
      let point = [];

      //  For each dimension in the data find the min/max range of that dimension,
      //  and choose a random value that lies within that range.
      for (let dimension = 0; dimension < dimensionality; dimension++) {
        const { min, max } = dimensionRanges[dimension];
        point[dimension] = min + Math.random() * (max - min);
      }
      centroids.push(point);
    }
    return centroids;
  }

  /**
   * Given a point in the data to consider, determine the closest
   * centroid and assign the point to that centroid.
   * The return value of this method is a boolean which represents
   * whether the point's centroid assignment has changed;
   * this is used to determine the termination condition for the algorithm.
   * @returns Did the point change its assignment?
   */
  assignPointToCentroid(pointIndex: number): boolean {
    const lastAssignedCentroid = this.centroidAssignments[pointIndex];
    const point = this.data[pointIndex];
    let minDistance: number | null = null;
    let assignedCentroid: number | null = null;
    for (let i = 0; i < this.centroids.length; i++) {
      const centroid = this.centroids[i];
      const distanceToCentroid = distance(point, centroid);
      if (minDistance === null || distanceToCentroid < minDistance) {
        minDistance = distanceToCentroid;
        assignedCentroid = i;
      }
    }
    this.centroidAssignments[pointIndex] = assignedCentroid!;
    return lastAssignedCentroid !== assignedCentroid;
  }

  /**
   * For all points in the data, call assignPointsToCentroids
   * and returns whether _any_ point's centroid assignment has
   * been updated.
   * @see assignPointToCentroid
   * @returns Was any point's centroid assignment updated?
   */
  assignPointsToCentroids(): boolean {
    let didAnyPointsGetReassigned = false;
    for (let i = 0; i < this.data.length; i++) {
      const wasReassigned = this.assignPointToCentroid(i);
      if (wasReassigned) didAnyPointsGetReassigned = true;
    }
    return didAnyPointsGetReassigned;
  }

  /**
   * Given a centroid to consider, returns an array
   * of all points assigned to that centroid.
   */
  getPointsForCentroid(centroidIndex: number) {
    const points = [];
    for (let i = 0; i < this.data.length; i++) {
      const assignment = this.centroidAssignments[i];
      if (assignment === centroidIndex) {
        points.push(this.data[i]);
      }
    }
    return points;
  }

  /**
   * Given a centroid to consider, update its location to
   * the mean value of the positions of points assigned to it.
   * @see getPointsForCentroid
   */
  updateCentroidLocation(centroidIndex: number) {
    const thisCentroidPoints = this.getPointsForCentroid(centroidIndex);
    const dimensionality = this.getDimensionality();
    const newCentroid = [];
    for (let dimension = 0; dimension < dimensionality; dimension++) {
      newCentroid[dimension] = mean(
        thisCentroidPoints.map(point => point[dimension])
      );
    }
    this.centroids[centroidIndex] = newCentroid;
    return newCentroid;
  }

  /**
   * For all centroids, call updateCentroidLocation
   */
  updateCentroidLocations() {
    for (let i = 0; i < this.centroids.length; i++) {
      this.updateCentroidLocation(i);
    }
  }

  /**
   * Calculates the total "error" for the current state
   * of centroid positions and assignments.
   * Here, error is defined as the root-mean-squared distance
   * of all points to their centroids.
   */
  calculateError() {
    let sumDistanceSquared = 0;
    for (let i = 0; i < this.data.length; i++) {
      const centroidIndex = this.centroidAssignments[i];
      const centroid = this.centroids[centroidIndex];
      const point = this.data[i];
      const thisDistance = distance(point, centroid) + this.k;
      sumDistanceSquared += thisDistance * thisDistance;
    }
    this.error = Math.sqrt(sumDistanceSquared / this.data.length);
    return this.error;
  }

  /**
   * Run the k-means algorithm until either the solver reaches steady-state,
   * or the maxIterations allowed has been exceeded.
   *
   * The return value from this method is an object with properties:
   * {
   * centroids {Array.<Object>},
   * iteration {number},
   * error {number},
   * didReachSteadyState {Boolean}
   * }
   *
   * You are most likely interested in the centroids property of the output.
   */
  solve(maxIterations = 1000) {
    while (this.iterations < maxIterations) {
      const didAssignmentsChange = this.assignPointsToCentroids();
      this.updateCentroidLocations();
      this.calculateError();
      this.iterationLogs[this.iterations] = {
        centroids: [...this.centroids],
        iteration: this.iterations,
        error: this.error,
        didReachSteadyState: !didAssignmentsChange,
      };

      if (didAssignmentsChange === false) {
        break;
      }

      this.iterations++;
    }
    return this.iterationLogs[this.iterationLogs.length - 1];
  }
}
