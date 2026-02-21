import tf from "@tensorflow/tfjs-node";

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential();

  // entrada de 11 posições (idade, cor, localização, renda, score, horas app, freq. viagem)
  // 100 neuronios

  model.add(
    tf.layers.dense({ inputShape: [11], units: 100, activation: "relu" }),
  );

  // Saída: 3 neuronios
  // um para cada categoria (premium, medium, basic)
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Treinamento do modelo
  await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 500,
    shuffle: true,
    callbacks: {
      // onEpochEnd: (epoch, log) => console.log(
      //     `Epoch: ${epoch}: loss = ${log.loss}`
      // )
    },
  });

  return model;
}

async function predict(model, pessoa) {
  const tfInput = tf.tensor2d([pessoa]);

  const pred = model.predict(tfInput);
  const predArray = await pred.array();
  return predArray[0].map((prob, index) => ({ prob, index }));
}

// Legenda das 11 colunas do Tensor:
// [0] Idade | [1] Azul | [2] Vermelho | [3] Verde | [4] SP | [5] RJ | [6] Curitiba
// [7] Renda | [8] Score | [9] Horas App | [10] Freq. Viagem

const tensorPessoasNormalizado = [
  [0.3, 1, 0, 0, 1, 0, 0, 0.45, 0.8, 0.2, 0.1], // Pessoa 1 (Premium provável)
  [0.15, 0, 1, 0, 0, 1, 0, 0.1, 0.4, 0.9, 0.05], // Pessoa 2 (Basic)
  [0.6, 0, 0, 1, 0, 0, 1, 0.55, 0.6, 0.3, 0.4], // Pessoa 3 (Medium)
  [0.45, 1, 0, 0, 1, 0, 0, 0.9, 0.95, 0.5, 0.8], // Pessoa 4 (Premium)
  [0.1, 0, 1, 0, 1, 0, 0, 0.05, 0.2, 0.8, 0.1], // Pessoa 5 (Basic)
  [0.8, 0, 0, 1, 0, 1, 0, 0.65, 0.75, 0.1, 0.2], // Pessoa 6 (Medium)
  [0.25, 1, 0, 0, 0, 0, 1, 0.3, 0.55, 0.4, 0.3], // Pessoa 7 (Basic)
  [0.5, 0, 1, 0, 1, 0, 0, 0.85, 0.9, 0.6, 0.75], // Pessoa 8 (Premium)
  [0.35, 0, 0, 1, 0, 1, 0, 0.4, 0.5, 0.5, 0.4], // Pessoa 9 (Medium)
  [0.2, 1, 0, 0, 0, 1, 0, 0.2, 0.35, 0.7, 0.15], // Pessoa 10 (Basic)
  [0.7, 0, 1, 0, 0, 0, 1, 0.75, 0.85, 0.25, 0.6], // Pessoa 11 (Premium)
  [0.4, 0, 0, 1, 1, 0, 0, 0.5, 0.65, 0.45, 0.35], // Pessoa 12 (Medium)
  [0.55, 1, 0, 0, 0, 1, 0, 0.95, 0.98, 0.15, 0.9], // Pessoa 13 (Premium)
  [0.05, 0, 1, 0, 0, 0, 1, 0.12, 0.3, 0.85, 0.05], // Pessoa 14 (Basic)
  [0.9, 0, 0, 1, 1, 0, 0, 0.48, 0.55, 0.2, 0.25], // Pessoa 15 (Medium)
];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"];
const tensorLabels = [
  [1, 0, 0], // Pessoa 1  - Premium (Renda/Score altos)
  [0, 0, 1], // Pessoa 2  - Basic (Renda baixa, muito uso de app)
  [0, 1, 0], // Pessoa 3  - Medium (Perfil equilibrado)
  [1, 0, 0], // Pessoa 4  - Premium
  [0, 0, 1], // Pessoa 5  - Basic
  [0, 1, 0], // Pessoa 6  - Medium
  [0, 0, 1], // Pessoa 7  - Basic
  [1, 0, 0], // Pessoa 8  - Premium
  [0, 1, 0], // Pessoa 9  - Medium
  [0, 0, 1], // Pessoa 10 - Basic
  [1, 0, 0], // Pessoa 11 - Premium
  [0, 1, 0], // Pessoa 12 - Medium
  [1, 0, 0], // Pessoa 13 - Premium
  [0, 0, 1], // Pessoa 14 - Basic
  [0, 1, 0], // Pessoa 15 - Medium
];

const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

const model = await trainModel(inputXs, outputYs);

// Perfil de teste Zé da Manga
const zeProfile = {
  nome: "Zé da Manga",
  idade: 28,
  cor: "verde",
  localizacao: "Curitiba",
  rendaMensal: "R$ 4.500",
  scoreCredito: 650,
  tempoUsoApp: "2h/dia",
  frequenciaViagem: "Baixa",
};

/**
 * REPRESENTAÇÃO NO TENSOR (INPUT):
 * [
 * 0.20, // Idade
 * 0,    // Cor: Azul
 * 0,    // Cor: Vermelho
 * 1,    // Cor: Verde
 * 0,    // Loc: São Paulo
 * 0,    // Loc: Rio de Janeiro
 * 1,    // Loc: Curitiba
 * 0.45, // Renda
 * 0.50, // Score
 * 0.30, // Horas App
 * 0.15  // Freq. Viagem
 * ]
 */
const personTensor = [0.2, 0, 0, 1, 0, 0, 1, 0.45, 0.5, 0.3, 0.15];

const predictions = await predict(model, personTensor);
const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map((p) => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
  .join("\n");
console.log(results);
