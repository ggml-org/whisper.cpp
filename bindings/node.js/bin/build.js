#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const https = require('https');
const { createWriteStream } = require('fs');

async function downloadAndUnzip(url, filename, output) {
	const file = createWriteStream(filename);

	console.log(`Downloading from ${url} ...`);
	await new Promise((resolve, reject) => {
		https.get(url, (response) => {
			if (response.statusCode === 302 && response.headers.location) {
				https.get(response.headers.location, (redirectResponse) => {
					redirectResponse.pipe(file);
					file.on('finish', resolve);
					file.on('error', reject);
				}).on('error', reject);
			} else {
				response.pipe(file);
				file.on('finish', resolve);
				file.on('error', reject);
			}
		}).on('error', reject);
	});
	console.log(`Download complete. Extracting...`);

	const unzipCmd = process.platform === 'win32' ? 
		`powershell -command "Expand-Archive -Path '${filename}' -DestinationPath '${output}' -Force"` :
		`unzip -o "${filename}" -d "${output}"`;
	execSync(unzipCmd, { stdio: 'inherit' });
	fs.unlinkSync(filename);
}

const isLatestRequested = process.argv[3] == 'latest';
const releaseUrl = isLatestRequested ? 'https://github.com/ggml-org/whisper.cpp/archive/refs/heads/master.zip' : 'https://github.com/ggml-org/whisper.cpp/archive/refs/tags/v1.8.3.zip';
const whisperCppDir = path.resolve(__dirname, '..', 'whisper.cpp');
const buildCmd = `cd ${whisperCppDir} && npx cmake-js clean && npx cmake-js compile -T addon.node -B Release`;

async function install() {
	console.log(
		'whisper.cpp will now be downloaded from the github repo and the addon will be built.'
		+ '\nIt is going to a take a while ...'
	);
	// extract to a .tmp directory because it's not yet know what directory the files will be extracted to
	const whisperCppTmpDir = whisperCppDir + '.tmp';
	await downloadAndUnzip(releaseUrl, path.resolve(__dirname, '..', releaseUrl.split('/').pop()), whisperCppTmpDir);
	// move the extracted files out from the first (and only) directory inside to "whisper.cpp"
	fs.renameSync(path.resolve(whisperCppTmpDir, fs.readdirSync(whisperCppTmpDir)[0]), whisperCppDir);
	fs.rmdirSync(whisperCppTmpDir);
	console.log(`Building ...`);
	execSync(buildCmd, { stdio: 'inherit' });
	console.log('whisper.cpp.node should now be ready for use.');
}

if (process.argv[2] == 'install') {
	if (isLatestRequested) {
		if (fs.existsSync(whisperCppDir)){
			console.log('Replacing with the latest one from master.');
			execSync(`rm -rf ${whisperCppDir}`, { stdio: 'inherit' });
		}
		install();
	} else if (fs.existsSync(whisperCppDir)) {
		console.log(
			'The directory with whisper.cpp exists so the addon should be already available for use.'
			+ '\nIf you think there\'s something wrong with the installation and would like to install afresh, then run again with the "reinstall" option.'
		);
	} else
		install();
} else if (process.argv[2] == 'reinstall') {
	// remove if already there and install
	execSync(`rm -rf ${whisperCppDir}`, { stdio: 'inherit' });
	install();
} else if (process.argv[2] == 'rebuild') {
	console.log(
		'whisper.cpp.node will now be rebuilt.'
		+ '\nIt is going to a take a while ...'
	);
	execSync(buildCmd, { stdio: 'inherit' });
	console.log('whisper.cpp.node should now be ready for use.');
} else
	console.log('Not sure what you want to do right now. Please pick one of "install", "reinstall" or "rebuild" commands.');
