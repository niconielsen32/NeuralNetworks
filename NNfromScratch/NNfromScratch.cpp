#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <functional>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

typedef vector<double> VD;
typedef vector<vector<double> > MD;




// First create an instance of an engine.
random_device rnd_device;
// Specify the engine and distribution.
mt19937 mersenne_engine {rnd_device()};  // Generates random integers
uniform_real_distribution<double> distribution {-0.5, 0.5};

auto randNum = [](){
    return distribution(mersenne_engine);
};



MD initLayerWeights(int inputs, int outputs){

    VD weights(inputs);
    MD layerWeights;

    for(int i = 0; i < outputs; i++){
        generate(weights.begin(), weights.end(), randNum);
        layerWeights.push_back(weights);
    }

    return layerWeights;
}



VD initLayerBiases(int outputs){

    VD biases(outputs);
    generate(biases.begin(), biases.end(), randNum);

    return biases;
}





int main()
{
    float inputsVec[] = {0,0,120,120,255,255,255,0,255,0,255,255};

    Mat inputs(4,3,CV_32F, inputsVec);
    Mat second;
    second = inputs;

    inputs /= 255;

    Mat flatten = inputs.reshape(1,1);

    cout << flatten << endl;

    //Mat result = inputs.mul(second);
    cout << "Input: " << inputs << endl;
    cout << "Second: " << second << endl;
    //cout << "Result: " << result << endl;




    MD weights1, weights2;
    VD biases1, biases2;

    weights1 = initLayerWeights(10, 10);
    biases1 = initLayerBiases(10);

    weights2 = initLayerWeights(10, 10);
    biases2 = initLayerBiases(10);


    return 0;
}
