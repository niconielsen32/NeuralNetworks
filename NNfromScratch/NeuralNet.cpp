#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <functional>
#include <vector>


using namespace std;


typedef vector<double> VD;
typedef vector<vector<double> > MD;

#include "matrixOperations.h"



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
    VD image = {10,40,30,3,255,220,0,7,8,186,4}; 

    for(auto &i : image){
        i /= 255;
    }

    MD weights1, weights2;
    VD biases1, biases2;

    weights1 = initLayerWeights(10, 10);
    biases1 = initLayerBiases(10);

    weights2 = initLayerWeights(10, 10);
    biases2 = initLayerBiases(10);



    for (size_t i = 0; i < weights2.size(); i++)
    {
        for (size_t j = 0; j < weights2[i].size(); j++)
        {
            cout << weights2[i][j] << " ";
        }
        cout << endl;
    }

    for(auto &i : biases1){
        cout << i << endl;
    }
    cout << biases1.size() << endl;




    



    
    return 0;
}
