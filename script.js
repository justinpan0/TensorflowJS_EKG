const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
const async = require('async')
const fs = require('fs');
const Papa = require('papaparse')

var inputFile = 'samples.csv';
var count = 0; // cache the running count
var train = [];
async function myEKGTrain(x_dat, y_dat) {
    const x = tf.tensor(x_dat);
    const y = tf.tensor(y_dat);

    const model = tf.sequential();

    const config_hidden = {
        inputShape:[count],
        activation: 'sigmoid',
        units: 1
    }
    const config_output={
        units:1,
        activation:'sigmoid'
    }

    const hidden = tf.layers.dense(config_hidden);
    const output = tf.layers.dense(config_output);

    model.add(hidden);
    model.add(output);

    const optimize=tf.train.sgd(0.1);

    const config={
        optimizer:optimize,
        loss:'meanSquaredError'
    }

    model.compile(config);

    await model.fit(x, y, { epochs: 250 });
    //await model.save("file://./my-model-1");
    console.log("Finish Training");
}


function parseData(input, callBack) {
    const file = fs.createReadStream(input);
    var data = [];
    var temp = [];
    var header = 2;

    Papa.parse(file, {
        worker: true, // Don't bog down the main thread if its a big file
        step: function (result) {
            if (header === 0) {
                temp.push(parseFloat(result.data[0][1]));
                count++;
            } else {
                header--;
            }
        },
        complete: function (results, file) {
            data.push(temp);
            console.log(data[0][0] + "," + data[0][1] + "\n....");
            console.log('parsing complete read', count, 'records.');
            callBack(data, [[1]]);
        }
    });
}

parseData(inputFile, myEKGTrain);
