---
layout: post
title: "Classification : Perceptron"
subtitle: "분류를 위한 가장 기초적인 방식인 퍼셉트론의 원리에 대해 알아봅니다."
date: 2026-11-12 14:34:00 +0900
categories: [인공지능]
tags: [인공지능, Classification, Perceptron]
---

# Classification : Perceptron

우리가 해결해야할 문제

To find a classification boundary

→ 이제 우리는 분류를 위한 최적의 분류선을 찾는 방법에 대해 배운다.

![image.png](image.png)

# Perceptron

$$
input(feature) :x = \begin{bmatrix} x_1 \\ ... \\ x_d \end{bmatrix} \\
weight : w = input :x = \begin{bmatrix} w_1 \\ ... \\ w_d \end{bmatrix} \\
$$

의사결정은 feature들의 단순 선형 결합을 기반으로 이루어집니다.

각 속성에 가중치를 곱해 모두 더한$\sum_{i=1}^d \omega_i x_i$로 표현 가능하다.

- 결정함수의 일반화

위의 과정을 부호(sign) 함수를 이용해 $g(x) = \text{sign}((\sum_{i=1}^d \omega_i x_i) - \text{threshold})$라는 하나의 결정 수식으로 정리합니다. 계산을 단순화하기 위해 식에 포함된 편향을 가중치 $w_0$로 치환하여 최종적으로 다음과 같은 표현식을 가집니다.

$$
w_0 = -threshold\\
g(x) = \text{sign}((\sum_{i=1}^d \omega_i x_i) + \omega_0)\\
g(x) >0: approve(+1) \\
g(x) < 0 : refuse(-1)
$$

==왜 Bias가 생길까?==

$input = 0$라면 경계에 있는 상황이므로 우리는 판별하기가 쉽지않다.

그렇기 때문에 가중치를 둬 경계에 두는 상황을 방지한다.

## Vectorization

인위적 좌표인 $x_0 = 1$을 도입합니다.

$$
g(x) = \text{sign}\left(\sum_{i=0}^d \omega_i x_i\right) = \text{sign}(\omega^T x)
$$

Classification boundary : hyperplane

*hyperplan(초평면) : 데이터를 두 그룹으로 나눌 떄 2차원 평면에서는 ‘직선’, 3차원 공간에서는 ‘평면’이 경계선 역할을 한다.

이를 확장하면 N차원 공간에 (N-1)차원의 경계면을 초평면이라 한다.

$$
\text hyperplane:g(x)= w^Tx = 0
$$

$w^T x = 0$을 만족하는 모든 벡터 $x$(즉, 초평면 위의 점들)는 가중치 벡터 $w$와 직교(Orthogonal)한다.

Blue points : $w^Tx > 0$

Green points : $w^Tx < 0$

## Perceptron Algotithm

perceptron implements

- $h(x) = sign(w^Tx)$

training set

- $(x_1,y_1),(x_2,y_2), ...,(x_N,y_N) \; \text where\; y_i \in \{-1, 1\}$

예측된 부호와 실제 정답이 다른, 즉 오분류된 포인트 $x_n$을 찾습니다.

→ $\text{sign}(\omega^T x_n) \neq y_n$

오분류된 데이터를 바탕으로 가중치 벡터를 업데이트 합니다.

→ $\omega \leftarrow \omega + y_n x_n$

- $y = +1$

![image.png](image%201.png)

![image.png](image%202.png)

![image.png](image%203.png)

- $y = -1$

![image.png](image%204.png)

![image.png](image%205.png)

![image.png](image%206.png)

## why perceptron updates work?

데이터 $x_n$이 잘못 분류된 상황으로 가정하자

→ $y_n=+1, sign(w^Tx_n) < 0 \;(negative)$

그러면 새로운 가중치를 업데이트 하자.

$\omega_{new} = \omega_{old} + y_n x_n = \omega_{old} + x_n\; (y_n= +1)$

$\omega_{new}^T x_n = (\omega_{old} + x_n)^T x_n = \omega_{old}^T x_n + x_n^T x_n$

기존 내적 값($w_{old}^Tx_n$)에 어떤 양수 값($x_n^Tx_n$)을 더한 형태

여기서, 자기 자신을 내적하면 무조건 양수

따라서 기울기는 음수에서 양수로 가까워지며 라벨에 따라 분류될 가능성이 높다.

## Diagram of Perceptron

![image.png](image%207.png)

update 규칙 2가지:

- 이산형 퍼셉트론 : $\omega \leftarrow \omega \pm 1 \cdot x_n$
- 연속형 경사 하강법 : $\omega \leftarrow \omega - \alpha\nabla_\omega f$

## Loss Function for perceptron

$$
L(w) = max\{0,\;-y_n \cdot(w^Tx_n)\}
$$

- $y_n \cdot (\omega^T x_n)$

정답과 예측의 부호가 같으면 두 값을 곱하면 항상 **양수**

부호가 다르면 두 값을 곱하면 **음수**

따라서 정답이라면 $Loss = 0$을 반환하며

오답인 경우 $Loss > 0$을 반환해 가중치 업데이트가 발생한다.

## Limitation

- 데이터가 두 클래스로 완벽히 나뉠 수 있는(Linearly separable) 상태라면 퍼셉트론은 두 클래스를 가르는 선(초평면, Hyperplane)을 찾을 수 있습니다.
- 하지만 퍼셉트론은 오분류가 0이 되는 순간 학습을 멈추기 때문에, **수많은 정답 선 중에서 '우연히 먼저 찾은 아무 선'** 하나만을 제시합니다.
- 첨부된 그래프의 검은색 선(perceptron)을 보면 빨간 점(C1)과 파란 점(C2)을 나누고는 있지만, 빨간 점들에 너무 아슬아슬하게 붙어 있습니다. 이는 새로운 데이터가 들어왔을 때 오분류할 위험이 매우 높은 불안정한 경계선입니다.

$\therefore$ ‘거리 정보’를 활용해 경계선이 양쪽 데이터들로부터 최대한의 마진을 확보해 지나가도록 한다.

→ SVM, Logistic Regression