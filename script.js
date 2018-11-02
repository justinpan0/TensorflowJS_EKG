require('@tensorflow/tfjs-node')
require('async')

const util = require('util')
const tf = require('@tensorflow/tfjs')
const fs = require('fs');

const readFile = util.promisify(fs.readFile);

var inputFile = ["data/arr_0.csv", "data/arr_1.csv", "data/arr_2.csv", "data/arr_3.csv", "data/arr_4.csv",
    "data/normal_0.csv", "data/normal_1.csv", "data/normal_2.csv", "data/normal_3.csv", "data/normal_4.csv"];

async function myEKGTrain(x_dat, y_dat) {
    const x = tf.tensor(x_dat);
    const y = tf.tensor(y_dat);
    console.log(x, y);
    console.log(x_dat[0].length);
    const model = tf.sequential();

    const config_hidden = {
        inputShape: [x_dat[0].length],
        activation: 'relu',
        units: 10
    }
    const config_output={
        units: 1,
        activation:'relu'
    }

    const hidden = tf.layers.dense(config_hidden);
    const output = tf.layers.dense(config_output);

    model.add(hidden);
    model.add(output);

    const optimize=tf.train.sgd(1e-4);

    const config={
        optimizer:optimize,
        loss:'meanSquaredError'
    }

    model.compile(config);

    await model.fit(x, y, { epochs: 250 });
    await model.save("file://./my-model-1");
    console.log("Finish Training");
}

async function read1(input) {
    var train_x = [];
    var train_y = [[1], [1], [1], [1], [1], [0], [0], [0], [0], [0]];

    for (let i = 0; i < input.length; i++) {
        var set = [];
        file = await readFile(input[i], 'utf8');
        data = file.split(/\n/);

        //for (let j = 2; j < data.length; j++) {
        for (let j = 2; j < 7679; j++) {
            // var temp = [];
            // console.log(parseFloat(data[j].split(",")[1]));
            // temp.push(parseFloat(data[j].split(",")[1]));
            // set.push(temp);
            set.push(parseFloat(data[j].split(",")[2]));
        }

        train_x.push(set);
    }

    myEKGTrain(train_x, train_y);
}

read1(inputFile);

/*
const Papa = require('papaparse')
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
*/
//parseData(inputFile, myEKGTrain);