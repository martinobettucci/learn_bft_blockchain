# Byzantium Attack Emulator with BFT Visualization

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Video Demo](#video-demo)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Set Up a Virtual Environment](#set-up-a-virtual-environment)
  - [Install Dependencies](#install-dependencies)
- [Usage](#usage)
  - [Running the Emulator](#running-the-emulator)
  - [Language Selection](#language-selection)
- [Internationalization (i18n)](#internationalization-i18n)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Description

Welcome to the **Byzantium Attack Emulator**! This application simulates Byzantine Fault Tolerance (BFT) mechanisms to demonstrate how they prevent system compromise in distributed networks. The emulator visualizes interactions between honest and malicious nodes, showcasing the resilience of BFT in maintaining network integrity.

## Features

- **Interactive Graph Visualization**: Real-time visualization of network nodes and their interactions.
- **Internationalization (i18n) Support**: Available in multiple languages including English, Italian, French, and Spanish.
- **Configurable Parameters**:
  - **Detection Threshold**: Adjust the sensitivity for detecting malicious nodes.
  - **System Compromise Threshold**: Define the proportion of malicious nodes that can compromise the system.
  - **Messages per Phase**: Set the number of messages exchanged in each simulation phase.
  - **Simulation Speed**: Control the speed of the simulation using a slider.
- **Detailed Logging Console**: Monitor simulation events and statuses.
- **Dynamic Legend**: Understand node types and message categories through an interactive legend.

## Video Demo

https://github.com/user-attachments/assets/41e75e72-3bf8-46fe-be49-64afbd27a469

## Installation

### Prerequisites

- **Python 3.6 or higher**: Ensure Python is installed on your system. You can download it from [Python's official website](https://www.python.org/downloads/).
- **Git**: Install Git to clone the repository. Download it from [Git's official website](https://git-scm.com/downloads).

### Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone git@github.com:martinobettucci/learn_bft_blockchain.git
cd learn_bft_blockchain
