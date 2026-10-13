const path = require('path');
const os = require('os');
const { promisify } = require('util');

const buildPath = os.platform() == 'win32' ? './whisper.cpp/build/bin/Release/addon.node' : './whisper.cpp/build/Release/addon.node';

const { whisper } = require(path.join(__dirname, buildPath));

module.exports = promisify(whisper);
