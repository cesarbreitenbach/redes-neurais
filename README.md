# Classificação de Planos com Inteligência Artificial

Este repositório contém uma implementação de uma Rede Neural Artificial utilizando **TensorFlow.js** em ambiente **Node.js**. O modelo categoriza usuários entre três tipos de planos (Premium, Medium e Basic) baseando-se em comportamentos e dados demográficos.

## Tecnologias

- **Node.js** (v22.10.0)
- **TensorFlow.js for Node** (@tensorflow/tfjs-node)

## O Problema

O sistema recebe 11 variáveis de entrada (features) que descrevem o perfil de um usuário:

- **Demográficas:** Idade e Localização (One-Hot Encoded).
- **Preferências:** Cor favorita (utilizada para testar a capacidade de filtragem da rede).
- **Financeiras:** Renda Mensal e Score de Crédito (Normalizados).
- **Comportamentais:** Tempo de uso do App e Frequência de viagens.

O objetivo é prever a probabilidade de o usuário pertencer a um dos três perfis de plano.

## Arquitetura do Modelo

O modelo foi construído utilizando a API Sequencial do TensorFlow:

1.  **Input Layer:** 11 dimensões.
2.  **Hidden Layer:** 100 neurônios com ativação **ReLU** (para capturar padrões não-lineares).
3.  **Output Layer:** 3 neurônios com ativação **Softmax**, entregando uma distribuição probabilística entre as categorias.

## Como Executar

1. Instale as dependências:
   ```bash
   npm install
   ```
2. rode o programa:
   ```bash
   npm start
   ```
