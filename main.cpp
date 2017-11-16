#include <QDateTime>
#include <QStringList>
#include <QTextStream>

#include <cstdint>
#include <cmath>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "random_generator.h"
#include "recursive_accumulator.h"

#include "input_vectors.h"    // input vectors
#include "target_size.h"      // output values

// if the output values represent compressed file sizes in bytes, the percentage error metric should be used (default),
// otherwise, the error metric is ordinary RMSE (comment line below)
#define FILE_SIZE

// learning method: use Adam (default) or gradient descent with momentum (comment line below)
#define USE_ADAM

int main(int argc, char *argv[])
{
    // note app start time to calculate total execution time
    const uint64_t appStartTime = QDateTime::currentMSecsSinceEpoch();

    QString appName = "mlp-standard";
    QString msgPrefix = '[' + appName + ']' + ' ';

    // ---------------------------------------------------------------------------------------------- //
    // -- parse command line arguments -------------------------------------------------------------- //
    // ---------------------------------------------------------------------------------------------- //

    // essential arguments: number of layers and neurons in them (excluding output neuron)
    uint64_t numLayers = 0;
    std::vector<uint64_t> numNeuronsInLayer;

    // optional arguments: max number of iterations, initial seed for random generator and number of CPU threads
    uint64_t seed = 0;
    uint64_t maxIterations = 10;
    uint64_t numThreads = 1;

    // for each argument
    for (int i = 1;  i < argc;  i++)
    {
        QString currentArgument (argv[i]);

        if (currentArgument == "-h" || currentArgument == "--help") {
            QTextStream(stdout, QIODevice::WriteOnly) << "syntax: " << appName << " -n <neurons_in_hidden_layer_1>,<neurons_in_hidden_layer_2>,...  [-s <initial_seed>] [-i <max_iterations>] [-t <cpu_threads>]" << '\n';
            return 0;
        } else if (currentArgument == "-n") {
            i++;
            if (i == argc) {
                QTextStream(stderr, QIODevice::WriteOnly) << msgPrefix << "error: no value(s) after \"-n\"" << '\n';
                return -1;
            }

            // split following string by commas
            QStringList listOfHiddenLayers = QString(argv[i]).split(QChar(','));

            // take into account input layer
            numLayers = listOfHiddenLayers.size() + 1;

            // reserve number for input layer
            numNeuronsInLayer.clear();
            numNeuronsInLayer.push_back(0);

            // parse each value
            for (uint64_t i = 1;  i < numLayers;  i++) {
                bool ok;
                numNeuronsInLayer.push_back(listOfHiddenLayers.at(i - 1).toUInt(&ok));
                if (!ok || numNeuronsInLayer[i] == 0) {
                    QTextStream(stderr, QIODevice::WriteOnly) << "error: can't parse value(s) after \"-n\", please use natural numbers" << '\n';
                    return -1;
                }
            }
            numNeuronsInLayer.shrink_to_fit();
        } else if (currentArgument == "-s") {
            i++;
            if (i == argc) {
                QTextStream(stderr, QIODevice::WriteOnly) << msgPrefix << "error: no value after \"-s\"" << '\n';
                return -1;
            }
            bool ok;
            seed = QString(argv[i]).toULongLong(&ok);
            if (!ok) {
                QTextStream(stderr, QIODevice::WriteOnly) << msgPrefix << "error: can't parse value after \"-s\"" << '\n';
                return -1;
            }
        } else if (currentArgument == "-i") {
            i++;
            if (i == argc) {
                QTextStream(stderr, QIODevice::WriteOnly) << msgPrefix << "error: no value after \"-i\"" << '\n';
                return -1;
            }
            bool ok;
            maxIterations = QString(argv[i]).toULongLong(&ok);
            if (!ok) {
                QTextStream(stderr, QIODevice::WriteOnly) << msgPrefix << "error: can't parse value after \"-i\"" << '\n';
                return -1;
            }
        } else if (currentArgument == "-t") {
            i++;
            if (i == argc) {
                QTextStream(stderr, QIODevice::WriteOnly) << msgPrefix << "error: no value after \"-t\"" << '\n';
                return -1;
            }
            bool ok;
            numThreads = QString(argv[i]).toULongLong(&ok);
            if (!ok) {
                QTextStream(stderr, QIODevice::WriteOnly) << msgPrefix << "error: can't parse value after \"-t\"" << '\n';
                return -1;
            }
        } else {
            QTextStream(stderr, QIODevice::WriteOnly) << msgPrefix << "error: unknown option \"" + currentArgument + "\"\n";
            return -1;
        }
    }

    // verify that user specified hidden layers
    if (numLayers <= 1) {
        QTextStream(stderr, QIODevice::WriteOnly) << "error: no hidden layers specified" << '\n';
        return -1;
    }

    // ---------------------------------------------------------------------------------------------- //
    // -- define training and validation sets ------------------------------------------------------- //
    // ---------------------------------------------------------------------------------------------- //

    // how many videos (training and validation samples) we are going to process?
    const uint64_t numSamples = 2500;
    const uint64_t numTrainingSamples = 1250;
    const uint64_t numValidationSamples = numSamples - numTrainingSamples;

    // how many predefined handcrafted inputs this model should have?
    const uint64_t inputVectorSize = 15 + 1;    // 15 features + video length (in extended experiments also +scale_factor +crf +preset_code)

    // set last value as the number of input neurons
    numNeuronsInLayer[0] = inputVectorSize;

    // ---------------------------------------------------------------------------------------------- //
    // -- standardize all predefined inputs; -------------------------------------------------------- //
    // -- this can be done in separate scope as we don't need to keep standardization parameters ---- //
    // ---------------------------------------------------------------------------------------------- //

    // array of standardized inputs
    double *standardizedInputVectors = new double [numSamples * inputVectorSize];

    {
        // initialize array of accumulators with zeroes
        double meanAccumulators [inputVectorSize] = {};

        // calculate mean values
        for (uint64_t i = 0;  i < numSamples;  i++) {
            for (uint64_t j = 0;  j < inputVectorSize;  j++) {
                meanAccumulators[j] += inputVectors[i * inputVectorSize + j];
            }
        }
        for (uint64_t j = 0;  j < inputVectorSize;  j++) {
            meanAccumulators[j] /= numSamples;
        }

        // initialize second array of accumulators
        double sdevAccumulators [inputVectorSize] = {};

        // calculate corrected standard deviation
        for (uint64_t i = 0;  i < numSamples;  i++) {
            for (uint64_t j = 0;  j < inputVectorSize;  j++) {
                const double deviation = meanAccumulators[j] - inputVectors[i * inputVectorSize + j];
                sdevAccumulators[j] += deviation * deviation;
            }
        }
        for (uint64_t j = 0;  j < inputVectorSize;  j++) {
            sdevAccumulators[j] = sqrt(sdevAccumulators[j] / (numSamples - 1));
        }

        // perform standardization
        for (uint64_t i = 0;  i < numSamples;  i++) {
            for (uint64_t j = 0;  j < inputVectorSize;  j++) {
                const uint64_t offset = i * inputVectorSize + j;
                standardizedInputVectors[offset] = (inputVectors[offset] - meanAccumulators[j]) / sdevAccumulators[j];
            }
        }

        // print input standardization parameters
        QTextStream console (stdout, QIODevice::WriteOnly);
        console << "Input mean values: [";
        for (uint64_t j = 0;  j < inputVectorSize;  j++) {
            console << QString("%1").arg(meanAccumulators[j], 0, 'g', 15);
            if (j == (inputVectorSize - 1))
                console << "]\n";
            else
                console << ", ";
        }
        console << "Input sdev values: [";
        for (uint64_t j = 0;  j < inputVectorSize;  j++) {
            console << QString("%1").arg(sdevAccumulators[j], 0, 'g', 15);
            if (j == (inputVectorSize - 1))
                console << "]\n";
            else
                console << ", ";
        }
    }

    // ---------------------------------------------------------------------------------------------- //
    // -- calculate standardization parameters for target output values; ---------------------------- //
    // -- on the contrary to inputs, there is no need to perform actual standardization ------------- //
    // ---------------------------------------------------------------------------------------------- //

    double meanTargetValue = 0;
    double sdevTargetValue = 0;

    {
        // mean value
        for (uint64_t i = 0;  i < numSamples;  i++) {
            #ifdef FILE_SIZE
            meanTargetValue += log(targetValues[i]);
            #else
            meanTargetValue += targetValues[i];
            #endif
        }
        meanTargetValue /= numSamples;

        // corrected standard deviation
        for (uint64_t i = 0;  i < numSamples;  i++) {
            #ifdef FILE_SIZE
            const double deviation = meanTargetValue - log(targetValues[i]);
            #else
            const double deviation = meanTargetValue - targetValues[i];
            #endif
            sdevTargetValue += deviation * deviation;
        }
        sdevTargetValue = sqrt(sdevTargetValue / (numSamples - 1));

        // print output standardization parameters
        QTextStream(stdout, QIODevice::WriteOnly) <<
            QString("Output mean/sdev values: [%1, %2]\n\n").arg(meanTargetValue, 0, 'g', 15).arg(sdevTargetValue, 0, 'g', 15);
    }

    // ---------------------------------------------------------------------------------------------- //
    // -- calculate how many weights are in the entire network -------------------------------------- //
    // -- and how many weights are up to each layer ------------------------------------------------- //
    // ---------------------------------------------------------------------------------------------- //

    // number of weights up to each hidden layer including those in current layer
    std::vector<uint64_t> numWeightsUpToLayer (numLayers);

    // input layer doesn't have weights
    numWeightsUpToLayer[0] = 0;

    // for each hidden layer
    for (uint64_t layer = 1;  layer < numLayers;  layer++) {
        const uint64_t numWeightsCurrentLayer = (1 + numNeuronsInLayer[layer - 1]) * numNeuronsInLayer[layer];
        numWeightsUpToLayer[layer] = numWeightsUpToLayer[layer - 1] + numWeightsCurrentLayer;
    }

    // number of weights before output neuron, or "output neuron weights offset"
    // (this is useful as output neuron is processed separately)
    const uint64_t numWeightsBeforeOutputNeuron = numWeightsUpToLayer[numLayers - 1];

    // number of weights exclusively in output neuron
    // (it's also useful for processing it separately)
    const uint64_t numWeightsInOutputNeuron = 1 + numNeuronsInLayer[numLayers - 1];

    // total number of weights in the entire network including output neuron
    const uint64_t numTotalWeights = numWeightsBeforeOutputNeuron + numWeightsInOutputNeuron;

    // ---------------------------------------------------------------------------------------------- //
    // -- initialize network weights ---------------------------------------------------------------- //
    // ---------------------------------------------------------------------------------------------- //

    //const double w [] = {};
    double *weights = new double [numTotalWeights];

    {
        RandomGenerator generator (seed);
        for (uint64_t i = 0;  i < numTotalWeights;  i++) {
            weights[i] = generator.fp64() * 0.2 - 0.1;
            //weights[i] = w[i];
        }
    }

    // ---------------------------------------------------------------------------------------------- //
    // -- define class Solution for storing the best result during training; ------------------------ //
    // -- criterion for the best result is the lowest error for both training and validation sets --- //
    // ---------------------------------------------------------------------------------------------- //

    class Solution
    {
    public:
        uint64_t iteration;

        uint64_t numWeights;
        std::vector<double> weights;

        double trainingError;
        double validationError;

        Solution(uint64_t numWeights) {
            this->iteration = 0;
            this->numWeights = numWeights;
            this->weights = std::vector<double>(numWeights, 0);
            this->trainingError = 1.0E+12;
            this->validationError = 1.0E+12;
        }

        inline bool smallerErrors(double trainingError, double validationError) const {
            return (this->trainingError > trainingError) && (this->validationError > validationError);
        }

        inline void setNewSolution(uint64_t iteration, const double *newWeights, double trainingError, double validationError) {
            this->iteration = iteration;
            for (uint64_t i = 0;  i < numWeights;  i++) this->weights[i] = newWeights[i];
            this->trainingError = trainingError;
            this->validationError = validationError;
        }
    };

    Solution bestSolution (numTotalWeights);

    // ---------------------------------------------------------------------------------------------- //
    // -- define a function, which is called from each thread to save the results; ------------------ //
    // -- this function has no locking mechanism and uses external arrays --------------------------- //
    // -- to store intermadiate results for training and validation samples ------------------------- //
    // ---------------------------------------------------------------------------------------------- //

    // arrays for squared relative correction of prediction
    double *trainingSqrErrors = new double [numTrainingSamples];
    double *validationSqrErrors = new double [numValidationSamples];

    // 2D array for training set network derivatives;
    // each row is a set of derivatives corresponding to a single training sample (video)
    double *allSampleDerivatives = new double [numTrainingSamples * numTotalWeights];

    auto submitResult = [numTotalWeights, trainingSqrErrors, validationSqrErrors, allSampleDerivatives] (uint64_t sampleNumber, double sqrError, const double *sampleDerivatives)
    {
        // check if we are submitting training or validation sample
        if (sampleDerivatives == nullptr) {
            // validation sample
            validationSqrErrors[sampleNumber - numTrainingSamples] = sqrError;
        } else {
            // training sample
            trainingSqrErrors[sampleNumber] = sqrError;

            // offset of the corresponding derivatives set in the array
            const uint64_t offset = sampleNumber * numTotalWeights;

            // for each derivative
            for (uint64_t i = 0;  i < numTotalWeights;  i++) {
                allSampleDerivatives[offset + i] = sampleDerivatives[i];
            }
        }
    };

    // ---------------------------------------------------------------------------------------------- //
    // -- prepare a function, which is called from each thread to obtain a training or  ------------- //
    // -- validation sample; it handles training iterations by making all threads wait until -------- //
    // -- last of them finishes processing its sample ----------------------------------------------- //
    // ---------------------------------------------------------------------------------------------- //

    // current iteration counter (this variable must be volatile as in each single case it's accessed it must be from memory)
    volatile uint64_t iteration = 0;

    // current video sample counter
    uint64_t sample = 0;

    // invalid sample number indicating end of current iteration
    const uint64_t invalidSample = UINT64_MAX;

    // number of idle threads waiting for other threads to finish at every iteration
    uint64_t numIdleThreads = 0;

    // mutex guarding entrance to this function
    std::mutex jobRequestMutex;

    // wait condition for all threads
    std::condition_variable newIterationCondition;

    // allocate array for current gradient
    double *gradient = new double [numTotalWeights];

    #ifdef USE_ADAM

    // allocate and set to zero momentum vector and cumulative squared gradient vector
    double *m = new double [numTotalWeights]();
    double *v = new double [numTotalWeights]();

    #else

    // last cumulative gradient used for GD with momentum and must be set to 0 before first iteration
    double *lastCumulativeGradient = new double [numTotalWeights]();

    #endif

    auto getJobSample = [=, &bestSolution, &iteration, &sample, &numIdleThreads, &jobRequestMutex, &newIterationCondition] ()
    {
        // make only one thread enter this function
        // by locking function access mutex with special wrapper 'unique_lock'
        std::unique_lock<std::mutex> lock(jobRequestMutex);

        // this GOTO label is used in two cases:
        // 1. current thread, which was waiting for others to finish, was awaken for the next iteration (and already posesses the mutex);
        // 2. current thread finished processing iteration and still posesses the mutex to start a next one;
        // (of course, I can replace this label with a while loop, but it actually makes code more obscure than a label)
        ENTRANCE_LABEL:

        // we make a copy of the current iteration number in case if we get a spurious wakeup;
        // I made 'iteration' variable volatile because compiler may keep the same value when thread sleeps, but actual value increases
        const uint64_t currentIteration = iteration;

        // check if we run out of video samples at the current iteration
        if (sample < numSamples)
        {
            // return current sample number
            const uint64_t currentSample = sample;
            sample++;
            return currentSample;    // mutex is unlocked automatically
        }
        else
        {
            // we run out of video samples at the current training iteration;
            // we need to wait for all threads to enter this function, and
            // then restart all of them for the next iteration

            // increase number of threads that finished their jobs at current iteration
            numIdleThreads++;

            // check if current thread, which entered this function was the last awaited for by others
            if (numIdleThreads == numThreads)
            {
                // in this case we need to perform all that bulky stuff at the end of each iteration

                // ---------------------------------------------------------------------------------------------- //
                // -- calculate training and validation errors -------------------------------------------------- //
                // ---------------------------------------------------------------------------------------------- //

                // initialize error accumulator
                RecursiveAccumulator errorAccumulator;

                // for each training sample
                for (uint64_t i = 0;  i < numTrainingSamples;  i++) errorAccumulator.add(trainingSqrErrors[i]);

                // training error
                const double trainingRMSE = sqrt(errorAccumulator.result() / numTrainingSamples);

                // reset error accumulator
                errorAccumulator.reset();

                // for each validation sample
                for (uint64_t i = 0;  i < numValidationSamples;  i++) errorAccumulator.add(validationSqrErrors[i]);

                // validation error
                const double validationRMSE = sqrt(errorAccumulator.result() / numValidationSamples);

                // update best solution if both errors are smaller
                if (bestSolution.smallerErrors(trainingRMSE, validationRMSE)) bestSolution.setNewSolution(iteration, weights, trainingRMSE, validationRMSE);

                // ---------------------------------------------------------------------------------------------- //
                // -- gradient averaging (+ gradient length) ---------------------------------------------------- //
                // ---------------------------------------------------------------------------------------------- //

                // special value for calculating exact RMSE (or RMS-RCoP) gradient
                const double gradientMultiplier = sdevTargetValue / (trainingRMSE * numTrainingSamples);

                // sum for calculating gradient length
                double sumSqrDerivatives = 0;

                // for each weight's derivative
                for (uint64_t i = 0;  i < numTotalWeights;  i++)
                {
                    // initialize accumulator over columns of sample derivatives
                    RecursiveAccumulator accumulator;

                    // for each training sample
                    for (uint64_t sample = 0;  sample < numTrainingSamples;  sample++) {
                        accumulator.add(allSampleDerivatives[sample * numTotalWeights + i]);
                    }

                    // average accumulator and use multiplier to get partial derivative
                    const double derivative = gradientMultiplier * accumulator.result();

                    // copy derivative to gradient array
                    gradient[i] = derivative;

                    // accumulate squared derivatives for gradient length
                    sumSqrDerivatives += derivative * derivative;
                }

                // finally, gradient length
                const double gradientLength = sqrt(sumSqrDerivatives);

                // ---------------------------------------------------------------------------------------------- //
                // -- print weights and gradient ---------------------------------------------------------------- //
                // ---------------------------------------------------------------------------------------------- //

                {
                    QTextStream console (stdout, QIODevice::WriteOnly);

                    // print errors
                    #ifdef FILE_SIZE
                    console << QString("[iteration %1] RMSRCoP = (%2 / %3)").arg(iteration).arg(trainingRMSE * 100, 0, 'f', 10).arg(validationRMSE * 100, 0, 'f', 10) << '\n';
                    #else
                    console << QString("[iteration %1] RMSE = (%2 / %3)").arg(iteration).arg(trainingRMSE, 0, 'f', 10).arg(validationRMSE, 0, 'f', 10) << '\n';
                    #endif

                    // print weights
                    console << "weights  = " << '[';
                    for (uint64_t i = 0;  i < numTotalWeights;  i++)
                    {
                        if (i != 0) console << ", ";
                        console << QString("%1").arg(weights[i], 0, 'f', 10);
                    }
                    console << ']' << '\n';

                    // print gradient
                    console << "gradient = " << '[';
                    for (uint64_t i = 0;  i < numTotalWeights;  i++)
                    {
                        if (i != 0) console << ", ";
                        console << QString("%1").arg(gradient[i], 0, 'f', 10);
                    }
                    console << ']' << '\n';

                    // print gradient length
                    console << QString("|gradient| = %1").arg(gradientLength, 0, 'f', 10) << '\n';

                    // new line
                    console << '\n';
                }

                // ---------------------------------------------------------------------------------------------- //
                // -- adjust weights ---------------------------------------------------------------------------- //
                // ---------------------------------------------------------------------------------------------- //

                #ifdef USE_ADAM
                {
                    // use adaptive moment estimation - "Adam";
                    // define constants from original paper
                    const double alpha = 0.001;
                    const double b1 = 0.9;
                    const double b2 = 0.999;
                    const double epsilon = 1.0E-08;

                    // each weight is updated individually
                    for (uint64_t i = 0;  i < numTotalWeights;  i++)
                    {
                        // just a partial derivative
                        const double derivative = gradient[i];

                        // update momentum
                        const double newDerivativeMomentum = b1 * m[i] + (1 - b1) * derivative;
                        m[i] = newDerivativeMomentum;

                        // update squared gradient
                        const double newSqrDerivative = b2 * v[i] + (1 - b2) * (derivative * derivative);
                        v[i] = newSqrDerivative;

                        // update corresponding weight
                        weights[i] -= alpha * newDerivativeMomentum / (sqrt(newSqrDerivative) + epsilon);
                    }
                }
                #else
                {
                    // gradient descent constants
                    const double gradientCoefficient = 0.01;
                    const double momentumCoefficient = 1 - gradientCoefficient;

                    // calculate cumulative gradient
                    for (uint64_t i = 0;  i < numTotalWeights;  i++) {
                        lastCumulativeGradient[i] = gradientCoefficient * gradient[i] + momentumCoefficient * lastCumulativeGradient[i];
                    }

                    // adjust network weights
                    for (uint64_t i = 0;  i < numTotalWeights;  i++) {
                        weights[i] -= lastCumulativeGradient[i];
                    }
                }
                #endif

                // ---------------------------------------------------------------------------------------------- //
                // -- check if this was the last iteration ------------------------------------------------------ //
                // ---------------------------------------------------------------------------------------------- //

                if (currentIteration == maxIterations) {
                    return invalidSample;    // mutex is unlocked automatically
                }

                // otherwise, increase iteration counter and reset to 0 other counters
                iteration++;
                sample = 0;
                numIdleThreads = 0;

                // wake up all other threads for next iteration
                newIterationCondition.notify_all();

                // since we are still posessing the mutex, GOTO entrance to obtain first sample
                goto ENTRANCE_LABEL;
            }
            else
            {
                // current thread is not the last one at current iteration;
                // check if current iteration is the very last, so the thread can finish
                if (currentIteration == maxIterations)
                {
                    return invalidSample;    // mutex is unlocked automatically
                }
                else
                {
                    // this thread needs to wait for other threads to finish;
                    // it's placed into while loop to avoid spurious wakeups;
                    // the wakeup condition is a different iteration number
                    while (currentIteration == iteration) {
                        newIterationCondition.wait(lock);
                    }

                    // new iteration has begun and we stll posess the mutex
                    goto ENTRANCE_LABEL;
                }
            }
        }

        // end of job providing function
    };

    // ---------------------------------------------------------------------------------------------- //
    // -- define a thread function, which calculates all derivatives for a single video sample; ----- //
    // -- the result for each sample is a squared error and a vector of derivatives ----------------- //
    // ---------------------------------------------------------------------------------------------- //

    auto calculateDerivatives = [=] ()
    {
        // ---------------------------------------------------------------------------------------------- //
        // -- create arrays to store intermediate neuron results; --------------------------------------- //
        // -- only input layer and hidden layers' results are stored here ------------------------------- //
        // ---------------------------------------------------------------------------------------------- //

        std::vector< std::vector<double> > neuronResult(numLayers);
        for (uint64_t layer = 0;  layer < numLayers;  layer++) {
            neuronResult[layer] = std::vector<double>(numNeuronsInLayer[layer]);
        }

        // ---------------------------------------------------------------------------------------------- //
        // -- create arrays to store intermediate partial derivatives of each hidden neuron output; ----- //
        // -- each neuron has a correspondning array of derivatives that include all weights up to ------ //
        // -- current layer and those in current layer as well ------------------------------------------ //
        // ---------------------------------------------------------------------------------------------- //

        std::vector< std::vector< std::vector<double> > > neuronDerivatives(numLayers);                   // entire network level
        for (uint64_t layer = 0;  layer < numLayers;  layer++) {
            neuronDerivatives[layer] = std::vector< std::vector<double> >(numNeuronsInLayer[layer]);      // single layer level
            for (uint64_t neuron = 0;  neuron < numNeuronsInLayer[layer];  neuron++) {
                neuronDerivatives[layer][neuron] = std::vector<double>(numWeightsUpToLayer[layer], 0);    // neuron level
            }
        }

        // create array to store all partial derivatives for the network output
        double *sampleDerivatives = new double [numTotalWeights];

        // ---------------------------------------------------------------------------------------------- //
        // -- infinite loop over samples regardless of current iteration -------------------------------- //
        // ---------------------------------------------------------------------------------------------- //

        while (true)
        {
            // obtain a sample to process
            const uint64_t sample = getJobSample();
            if (sample == invalidSample) break;

            // check if this is a training sample
            const bool isTrainingSample = (sample < numTrainingSamples);

            // ---------------------------------------------------------------------------------------------- //
            // -- calculate network result ------------------------------------------------------------------ //
            // ---------------------------------------------------------------------------------------------- //

            // copy input vector as a result of input layer
            for (uint64_t i = 0;  i < inputVectorSize;  i++) {
                neuronResult[0][i] = standardizedInputVectors[sample * inputVectorSize + i];
            }

            // process hidden layers in a shadow scope
            {
                // temporary weights counter for entire network
                uint64_t counter = 0;

                // for each hidden layer
                for (uint64_t layer = 1;  layer < numLayers;  layer++)
                {
                    // extract number of neurons in this layer
                    const uint64_t numNeurons = numNeuronsInLayer[layer];

                    // extract number of inputs for neurons in this layer
                    const uint64_t numInputs = numNeuronsInLayer[layer - 1];

                    // for each neuron in the current layer
                    for (uint64_t neuron = 0;  neuron < numNeurons;  neuron++)
                    {
                        // initialize linear combination with bias
                        double linearCombination = weights[counter++];

                        // for each input of current neuron
                        for (uint64_t i = 0;  i < numInputs;  i++) {
                            linearCombination += weights[counter++] * neuronResult[layer - 1][i];
                        }

                        // apply exp sigmoid
                        const double result = 1 / (1 + exp(-linearCombination));

                        // save neuron result
                        neuronResult[layer][neuron] = result;
                    }
                }
            }

            // calculate entire network result;
            // initialize result with output neuron bias
            double networkResult = weights[numWeightsBeforeOutputNeuron];

            // for each input weight of the output neuron
            for (uint64_t i = 0;  i < (numWeightsInOutputNeuron - 1);  i++) {
                networkResult += weights[numWeightsBeforeOutputNeuron + 1 + i] * neuronResult[numLayers - 1][i];
            }

            #ifdef FILE_SIZE

            // de-standardize and exponentiate the result to get bytes
            const double predictedValue = exp(networkResult * sdevTargetValue + meanTargetValue);

            // get target value
            const double targetValue = targetValues[sample];

            // calculate relative correction of prediction
            const double RCoP = targetValue / predictedValue - 1;

            #else

            // de-standardize result to get predicted value
            const double predictedValue = networkResult * sdevTargetValue + meanTargetValue;

            // get target value
            const double targetValue = targetValues[sample];

            // prediction error
            const double predictionError = predictedValue - targetValue;

            #endif

            // ---------------------------------------------------------------------------------------------- //
            // -- calculate current sample derivatives for entire network ----------------------------------- //
            // ---------------------------------------------------------------------------------------------- //

            if (isTrainingSample)
            {
                // processing hidden layers first;
                // to store intermediate derivatives we use 'neuronDerivatives' array

                // for each hidden layer
                for (uint64_t layer = 1;  layer < numLayers;  layer++)
                {
                    // extract number of neurons in this layer
                    const uint64_t numNeurons = numNeuronsInLayer[layer];

                    // extract number of inputs neurons in this layer
                    const uint64_t numInputs = numNeuronsInLayer[layer - 1];

                    // extract offset of the current layer weights in the global weights array
                    const uint64_t layerWeightsOffset = numWeightsUpToLayer[layer - 1];

                    // for each neuron in the current layer
                    for (uint64_t neuron = 0;  neuron < numNeurons;  neuron++)
                    {
                        // calculate derivative of the sigmoid function of the current neuron
                        const double currentNeuronResult = neuronResult[layer][neuron];
                        const double sigmDerivative = currentNeuronResult * (1 - currentNeuronResult);

                        // calculate offset of the current neuron weights in the global weights array
                        const uint64_t neuronWeightsOffset = layerWeightsOffset + neuron * (1 + numInputs);

                        // for each weight's derivative in previous layers
                        for (uint64_t derivativeCounter = 0;  derivativeCounter < layerWeightsOffset;  derivativeCounter++)
                        {
                            // here we calculate a partial derivative of the linear combination of the current neuron
                            // by one of the weights in previous layers

                            // bias derivative is always 0
                            double derivative = 0;

                            // for each input of current neuron
                            for (uint64_t i = 0;  i < numInputs;  i++) {
                                derivative += weights[neuronWeightsOffset + 1 + i] * neuronDerivatives[layer - 1][i][derivativeCounter];
                            }

                            // save this partial derivative multiplied by sigmoid derivative
                            neuronDerivatives[layer][neuron][derivativeCounter] = derivative * sigmDerivative;
                        }

                        // time to calculate derivatives by weights of the current neuron;
                        // we don't have to zero all the derivatives by remaining weights in current layer,
                        // because all intermediate derivatives were initialized with zero,
                        // and neighbour neuron derivatives remain unchanged when processing current neuron

                        // derivative of the current neuron bias
                        neuronDerivatives[layer][neuron][neuronWeightsOffset] = sigmDerivative;

                        // derivatives of the input weights in current neuron
                        for (uint64_t i = 0;  i < numInputs;  i++) {
                            neuronDerivatives[layer][neuron][neuronWeightsOffset + 1 + i] = neuronResult[layer - 1][i] * sigmDerivative;
                        }

                        // next neuron
                    }

                    // next layer
                }

                // processing a single output neuron;
                // result is all partial derivatives for current sample, which are saved into 'sampleDerivatives' array
                {
                    // firstly, we calculate a weighted sum of derivatives by each weight in previous layers;
                    // secondly, using the same derivative counter, we find derivatives by output neuron weights;
                    // thirdly, we multiply all obtained derivatives by a special "sample derivative coefficient",
                    // which in case of RMSE is just a prediction error

                    // common counter
                    uint64_t derivativeCounter = 0;

                    // for each weight's derivative in previous layers
                    for (;  derivativeCounter < numWeightsBeforeOutputNeuron;  derivativeCounter++)
                    {
                        // output bias doesn't take part in the linear combination of derivatives
                        double derivative = 0;

                        // for each input of the output neuron
                        for (uint64_t i = 0;  i < (numWeightsInOutputNeuron - 1);  i++) {
                            derivative += weights[numWeightsBeforeOutputNeuron + 1 + i] * neuronDerivatives[numLayers - 1][i][derivativeCounter];
                        }

                        // save current partial derivative to the 'sampleDerivatives' array
                        sampleDerivatives[derivativeCounter] = derivative;
                    }

                    // derivative by the output neuron bias
                    sampleDerivatives[derivativeCounter++] = 1;

                    // derivatives by each input weight of the output neuron equal to the respective result of the last hidden layer neuron
                    for (uint64_t i = 0;  i < (numWeightsInOutputNeuron - 1);  i++) {
                        sampleDerivatives[derivativeCounter++] = neuronResult[numLayers - 1][i];
                    }

                    #ifdef FILE_SIZE

                    // finally, we need to apply our special derivative multiplier
                    const double sampleDerivativeMultiplier = RCoP * (-1) * targetValue / predictedValue;

                    for (uint64_t d = 0;  d < numTotalWeights;  d++) {
                        sampleDerivatives[d] *= sampleDerivativeMultiplier;
                    }

                    #else

                    // finally, we need to apply our special derivative multiplier
                    for (uint64_t d = 0;  d < numTotalWeights;  d++) {
                        sampleDerivatives[d] *= predictionError;
                    }

                    #endif
                }

                // end of derivatives calculation
            }

            // save results
            #ifdef FILE_SIZE
            submitResult(sample, RCoP * RCoP, (isTrainingSample ? sampleDerivatives : nullptr));
            #else
            submitResult(sample, predictionError * predictionError, (isTrainingSample ? sampleDerivatives : nullptr));
            #endif

            // next sample
        }

        // cleaning
        delete [] sampleDerivatives;

        // thread terminated
    };

    // ---------------------------------------------------------------------------------------------- //
    // -- actual training --------------------------------------------------------------------------- //
    // ---------------------------------------------------------------------------------------------- //

    std::thread threads [numThreads];
    for (uint64_t i = 0;  i < numThreads;  i++) threads[i] = std::thread(calculateDerivatives);
    for (uint64_t i = 0;  i < numThreads;  i++) threads[i].join();

    // ---------------------------------------------------------------------------------------------- //
    // -- print best solution, execution time etc. -------------------------------------------------- //
    // ---------------------------------------------------------------------------------------------- //

    // cleaning
    delete [] standardizedInputVectors;
    delete [] weights;

    delete [] trainingSqrErrors;
    delete [] validationSqrErrors;
    delete [] allSampleDerivatives;

    delete [] gradient;

    #ifdef USE_ADAM
    delete [] m;
    delete [] v;
    #else
    delete [] lastCumulativeGradient;
    #endif

    // print best weights
    QTextStream console (stdout, QIODevice::WriteOnly);
    #ifdef FILE_SIZE
    console << "Best solution RMSRCoP = (" << (bestSolution.trainingError * 100) << " / " << (bestSolution.validationError * 100) << ") was obtained at iteration " << bestSolution.iteration << ":\n";
    #else
    console << "Best solution RMSE = (" << (bestSolution.trainingError) << " / " << (bestSolution.validationError) << ") was obtained at iteration " << bestSolution.iteration << ":\n";
    #endif
    console << '[';
    for (uint64_t i = 0;  i < numTotalWeights;  i++)
    {
        if (i != 0) console << ", ";
        console << QString("%1").arg(bestSolution.weights[i], 0, 'g', 15);
    }
    console << ']' << '\n';

    // print total app execution time
    const uint64_t appEndTime = QDateTime::currentMSecsSinceEpoch();
    console << QString("Total execution time: %1 sec").arg((appEndTime - appStartTime) / 1000.0, 0, 'f', 3) << '\n';

    return 0;
}
