---
layout: ../layouts/BaseLayout.astro
title: "主页"
---

欢迎来到我的个人博客。这里记录我如何拆解问题、验证假设，再重建思考路径。

<section class="go-chain">
  <div class="go-chain__intro">
    <h2>围棋迷你对弈</h2>
    <p>体验类似ぷに碁的简化规则：轮流落子，吃掉无气的棋子，双方连续停着则结算。</p>
    <div class="go-chain__controls">
      <label class="go-chain__select">
        棋盘
        <select class="go-chain__size">
          <option value="5">5路盘</option>
          <option value="6">6路盘</option>
          <option value="7" selected>7路盘</option>
          <option value="8">8路盘</option>
          <option value="9">9路盘</option>
        </select>
      </label>
      <button class="go-chain__pass" type="button">停一手</button>
      <button class="go-chain__reset" type="button">重新开始</button>
      <span class="go-chain__status" aria-live="polite"></span>
    </div>
  </div>
  <div class="go-chain-board" role="grid" aria-label="围棋棋盘"></div>
  <div class="go-chain__score" aria-live="polite">
    <span class="go-chain__score-item">黑子：<strong class="go-chain__score-black">0</strong></span>
    <span class="go-chain__score-item">白子：<strong class="go-chain__score-white">0</strong></span>
  </div>
</section>

<script type="module">
  const boardElement = document.querySelector(".go-chain-board");
  const statusElement = document.querySelector(".go-chain__status");
  const resetButton = document.querySelector(".go-chain__reset");
  const passButton = document.querySelector(".go-chain__pass");
  const sizeSelect = document.querySelector(".go-chain__size");
  const blackScoreElement = document.querySelector(".go-chain__score-black");
  const whiteScoreElement = document.querySelector(".go-chain__score-white");

  const cells = [];
  let boardSize = Number(sizeSelect.value);
  let board = [];
  let currentPlayer = "black";
  let consecutivePasses = 0;

  const getNeighbors = (row, col) => [
    [row - 1, col],
    [row + 1, col],
    [row, col - 1],
    [row, col + 1],
  ];

  const getGroup = (row, col, color) => {
    const queue = [[row, col]];
    const visited = new Set([`${row},${col}`]);
    const group = [];

    while (queue.length) {
      const [currentRow, currentCol] = queue.shift();
      group.push([currentRow, currentCol]);

      for (const [nextRow, nextCol] of getNeighbors(currentRow, currentCol)) {
        if (
          nextRow < 0 ||
          nextRow >= boardSize ||
          nextCol < 0 ||
          nextCol >= boardSize
        ) {
          continue;
        }

        if (board[nextRow][nextCol] !== color) {
          continue;
        }

        const key = `${nextRow},${nextCol}`;
        if (visited.has(key)) {
          continue;
        }

        visited.add(key);
        queue.push([nextRow, nextCol]);
      }
    }

    return group;
  };

  const countLiberties = (group) => {
    const liberties = new Set();

    for (const [row, col] of group) {
      for (const [nextRow, nextCol] of getNeighbors(row, col)) {
        if (
          nextRow < 0 ||
          nextRow >= boardSize ||
          nextCol < 0 ||
          nextCol >= boardSize
        ) {
          continue;
        }

        if (!board[nextRow][nextCol]) {
          liberties.add(`${nextRow},${nextCol}`);
        }
      }
    }

    return liberties.size;
  };

  const updateBoard = () => {
    for (const cell of cells) {
      cell.classList.remove(
        "black",
        "white",
        "merge-up",
        "merge-down",
        "merge-left",
        "merge-right",
        "link-up-left",
        "link-up-right",
        "link-down-left",
        "link-down-right"
      );
    }

    for (let row = 0; row < boardSize; row += 1) {
      for (let col = 0; col < boardSize; col += 1) {
        const color = board[row][col];
        if (!color) {
          continue;
        }

        const cell = cells[row * boardSize + col];
        cell.classList.add(color);

        const hasUp = row > 0 && board[row - 1][col] === color;
        const hasDown = row < boardSize - 1 && board[row + 1][col] === color;
        const hasLeft = col > 0 && board[row][col - 1] === color;
        const hasRight = col < boardSize - 1 && board[row][col + 1] === color;

        if (hasUp) {
          cell.classList.add("merge-up");
        }
        if (hasDown) {
          cell.classList.add("merge-down");
        }
        if (hasLeft) {
          cell.classList.add("merge-left");
        }
        if (hasRight) {
          cell.classList.add("merge-right");
        }

        // ponytail: diagonal corner fill only for concave corners of a
        // same-color blob — require an orthogonal bridge, else isolated
        // diagonal stones poke spikes at each other.
        if (
          row > 0 &&
          col > 0 &&
          board[row - 1][col - 1] === color &&
          (board[row - 1][col] === color || board[row][col - 1] === color)
        ) {
          cell.classList.add("link-up-left");
        }
        if (
          row > 0 &&
          col < boardSize - 1 &&
          board[row - 1][col + 1] === color &&
          (board[row - 1][col] === color || board[row][col + 1] === color)
        ) {
          cell.classList.add("link-up-right");
        }
        if (
          row < boardSize - 1 &&
          col > 0 &&
          board[row + 1][col - 1] === color &&
          (board[row + 1][col] === color || board[row][col - 1] === color)
        ) {
          cell.classList.add("link-down-left");
        }
        if (
          row < boardSize - 1 &&
          col < boardSize - 1 &&
          board[row + 1][col + 1] === color &&
          (board[row + 1][col] === color || board[row][col + 1] === color)
        ) {
          cell.classList.add("link-down-right");
        }
      }
    }
  };

  const updateScores = () => {
    let blackScore = 0;
    let whiteScore = 0;

    for (let row = 0; row < boardSize; row += 1) {
      for (let col = 0; col < boardSize; col += 1) {
        if (board[row][col] === "black") {
          blackScore += 1;
        }
        if (board[row][col] === "white") {
          whiteScore += 1;
        }
      }
    }

    blackScoreElement.textContent = String(blackScore);
    whiteScoreElement.textContent = String(whiteScore);
  };

  const updateStatus = (message = "") => {
    const playerLabel = currentPlayer === "black" ? "黑子" : "白子";
    const info = message ? `｜${message}` : "";
    statusElement.textContent = `当前执子：${playerLabel}${info}`;
  };

  const switchPlayer = () => {
    currentPlayer = currentPlayer === "black" ? "white" : "black";
    updateStatus();
  };

  const captureGroup = (group) => {
    for (const [row, col] of group) {
      board[row][col] = "";
    }
  };

  const placeStone = (row, col) => {
    if (board[row][col]) {
      return;
    }

    board[row][col] = currentPlayer;
    const opponent = currentPlayer === "black" ? "white" : "black";
    let captured = false;
    const mergeTargets = new Set([row * boardSize + col]);

    for (const [nextRow, nextCol] of getNeighbors(row, col)) {
      if (
        nextRow < 0 ||
        nextRow >= boardSize ||
        nextCol < 0 ||
        nextCol >= boardSize
      ) {
        continue;
      }

      if (board[nextRow][nextCol] === currentPlayer) {
        mergeTargets.add(nextRow * boardSize + nextCol);
        continue;
      }

      if (board[nextRow][nextCol] === opponent) {
        const opponentGroup = getGroup(nextRow, nextCol, opponent);
        if (countLiberties(opponentGroup) === 0) {
          captureGroup(opponentGroup);
          captured = true;
        }
      }
    }

    const ownGroup = getGroup(row, col, currentPlayer);
    if (!captured && countLiberties(ownGroup) === 0) {
      board[row][col] = "";
      updateStatus("这里没有气，不能自杀");
      return;
    }

    consecutivePasses = 0;
    updateBoard();
    for (const index of mergeTargets) {
      const cell = cells[index];
      cell.classList.add("merge-animate");
    }
    setTimeout(() => {
      for (const index of mergeTargets) {
        const cell = cells[index];
        cell.classList.remove("merge-animate");
      }
    }, 240);
    updateScores();
    switchPlayer();
  };

  const createBoard = () => {
    boardElement.style.setProperty("--board-size", boardSize);
    boardElement.innerHTML = "";
    cells.length = 0;

    for (let row = 0; row < boardSize; row += 1) {
      for (let col = 0; col < boardSize; col += 1) {
        const cell = document.createElement("button");
        cell.type = "button";
        cell.className = "go-chain-cell";
        cell.dataset.row = String(row);
        cell.dataset.col = String(col);

        const stone = document.createElement("span");
        stone.className = "stone";
        cell.appendChild(stone);

        ["up", "down", "left", "right"].forEach((direction) => {
          const merge = document.createElement("span");
          merge.className = `merge ${direction}`;
          cell.appendChild(merge);
        });

        ["up-left", "up-right", "down-left", "down-right"].forEach((direction) => {
          const link = document.createElement("span");
          link.className = `link ${direction}`;
          cell.appendChild(link);
        });

        cell.addEventListener("click", () => {
          placeStone(row, col);
        });

        boardElement.appendChild(cell);
        cells.push(cell);
      }
    }
  };

  const resetGame = () => {
    board = Array.from({ length: boardSize }, () =>
      Array.from({ length: boardSize }, () => "")
    );
    currentPlayer = "black";
    consecutivePasses = 0;
    updateBoard();
    updateScores();
    updateStatus();
  };

  const passTurn = () => {
    consecutivePasses += 1;
    if (consecutivePasses >= 2) {
      updateScores();
      const blackScore = Number(blackScoreElement.textContent);
      const whiteScore = Number(whiteScoreElement.textContent);
      if (blackScore > whiteScore) {
        updateStatus("双方停着，黑子领先");
      } else if (whiteScore > blackScore) {
        updateStatus("双方停着，白子领先");
      } else {
        updateStatus("双方停着，平局");
      }
      return;
    }

    switchPlayer();
    updateStatus("停一手");
  };

  sizeSelect.addEventListener("change", () => {
    boardSize = Number(sizeSelect.value);
    createBoard();
    resetGame();
  });

  passButton.addEventListener("click", passTurn);
  resetButton.addEventListener("click", () => {
    createBoard();
    resetGame();
  });

  createBoard();
  resetGame();
</script>

<ul class="listing">
  <li>
    <strong><a href="blog/my-method/">我的方法：从拆解到重建</a></strong>
    <span class="meta">2025-01-01</span>
  </li>
  <li>
    <strong><a href="blog/speculative-decoding/">Speculative Decoding 读书笔记</a></strong>
    <span class="meta">2025-01-11</span>
  </li>
</ul>
