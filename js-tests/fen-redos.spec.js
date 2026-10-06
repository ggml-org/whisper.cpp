/* eslint-env mocha */
const { expect } = require('chai');
const { performance } = require('perf_hooks');
const path = require('path');

// —— 最小浏览器 & jQuery stub，避免 DOM 依赖 ——
global.window = global;
global.document = {};
global.jQuery = function(){};
global.$ = global.jQuery;


const target = path.join(
  __dirname,
  '..',
  'examples/wchess/wchess.wasm/chessboardjs-1.0.0/js/chessboard-1.0.0.js'
);

// 载入真实实现（会把 Chessboard 挂到全局）
require(target);
const Chessboard = global.Chessboard;

describe('FEN sanitize ReDoS in whisper.cpp (fen = fen.replace(/ .+$/, \'\'))', function () {
  this.timeout(60_000);

  it('should complete within 2 seconds', function () {
    const N = 100000; 
    const attack = ' '.repeat(N) + '\n@';

    const t0 = performance.now();
    try { Chessboard.fenToObj(attack); } catch (_) {}
    const ms = performance.now() - t0;


    expect(ms).to.be.lessThan(2_000);
  });
});




