// Common Javascript functions used by the examples

function convertTypedArray(src, type) {
    var buffer = new ArrayBuffer(src.byteLength);
    var baseView = new src.constructor(buffer).set(src);
    return new type(buffer);
}

var printTextarea = (function() {
    var element = document.getElementById('output');
    if (element) element.value = ''; // clear browser cache
    return function(text) {
        if (arguments.length > 1) text = Array.prototype.slice.call(arguments).join(' ');
        console.log(text);
        if (element) {
            element.value += text + "\n";
            element.scrollTop = element.scrollHeight; // focus on bottom
        }
    };
})();

async function clearCache() {
    if (confirm('Are you sure you want to clear the cache?\nAll the models will be downloaded again.')) {
        indexedDB.deleteDatabase(dbName);
        location.reload();
    }
}

// fetch a remote file from remote URL using the Fetch API
async function fetchRemote(url, cbProgress, cbPrint) {
    cbPrint('fetchRemote: downloading with fetch()...');

    const response = await fetch(
        url,
        {
            method: 'GET',
        }
    );

    if (!response.ok) {
        cbPrint('fetchRemote: failed to fetch ' + url);
        return;
    }

    const contentLength = response.headers.get('content-length');
    const total = parseInt(contentLength, 10);
    const reader = response.body.getReader();

    var chunks = [];
    var receivedLength = 0;
    var progressLast = -1;

    while (true) {
        const { done, value } = await reader.read();

        if (done) {
            break;
        }

        chunks.push(value);
        receivedLength += value.length;

        if (contentLength) {
            cbProgress(receivedLength/total);

            var progressCur = Math.round((receivedLength / total) * 10);
            if (progressCur != progressLast) {
                cbPrint('fetchRemote: fetching ' + 10*progressCur + '% ...');
                progressLast = progressCur;
            }
        }
    }

    var position = 0;
    var chunksAll = new Uint8Array(receivedLength);

    for (var chunk of chunks) {
        chunksAll.set(chunk, position);
        position += chunk.length;
    }

    return chunksAll;
}

// load remote data
// - check if the data is already in the IndexedDB
// - if not, fetch it from the remote URL and store it in the IndexedDB
function loadRemote(url, dst, size_mb, cbProgress, cbReady, cbCancel, cbPrint) {
    if (!navigator.storage || !navigator.storage.estimate) {
        cbPrint('loadRemote: navigator.storage.estimate() is not supported');
    } else {
        // query the storage quota and print it
        navigator.storage.estimate().then(function (estimate) {
            cbPrint('loadRemote: storage quota: ' + estimate.quota + ' bytes');
            cbPrint('loadRemote: storage usage: ' + estimate.usage + ' bytes');
        });
    }

    // check if the data is already in the IndexedDB
    var rq = indexedDB.open(dbName, dbVersion);

    rq.onupgradeneeded = function (event) {
        var db = event.target.result;
        if (db.version == 1) {
            var os = db.createObjectStore('models', { autoIncrement: false });
            cbPrint('loadRemote: created IndexedDB ' + db.name + ' version ' + db.version);
        } else {
            // clear the database
            var os = event.currentTarget.transaction.objectStore('models');
            os.clear();
            cbPrint('loadRemote: cleared IndexedDB ' + db.name + ' version ' + db.version);
        }
    };

    rq.onsuccess = function (event) {
        var db = event.target.result;
        var tx = db.transaction(['models'], 'readonly');
        var os = tx.objectStore('models');
        var rq = os.get(url);

        rq.onsuccess = function (event) {
            if (rq.result) {
                cbPrint('loadRemote: "' + url + '" is already in the IndexedDB');
                cbReady(dst, rq.result);
            } else {
                // data is not in the IndexedDB
                cbPrint('loadRemote: "' + url + '" is not in the IndexedDB');

                // alert and ask the user to confirm
                if (!confirm(
                    'You are about to download ' + size_mb + ' MB of data.\n' +
                    'The model data will be cached in the browser for future use.\n\n' +
                    'Press OK to continue.')) {
                    cbCancel();
                    return;
                }

                fetchRemote(url, cbProgress, cbPrint).then(function (data) {
                    if (data) {
                        // store the data in the IndexedDB
                        var rq = indexedDB.open(dbName, dbVersion);
                        rq.onsuccess = function (event) {
                            var db = event.target.result;
                            var tx = db.transaction(['models'], 'readwrite');
                            var os = tx.objectStore('models');

                            var rq = null;
                            try {
                                var rq = os.put(data, url);
                            } catch (e) {
                                cbPrint('loadRemote: failed to store "' + url + '" in the IndexedDB: \n' + e);
                                cbCancel();
                                return;
                            }

                            rq.onsuccess = function (event) {
                                cbPrint('loadRemote: "' + url + '" stored in the IndexedDB');
                                cbReady(dst, data);
                            };

                            rq.onerror = function (event) {
                                cbPrint('loadRemote: failed to store "' + url + '" in the IndexedDB');
                                cbCancel();
                            };
                        };
                    }
                });
            }
        };

        rq.onerror = function (event) {
            cbPrint('loadRemote: failed to get data from the IndexedDB');
            cbCancel();
        };
    };

    rq.onerror = function (event) {
        cbPrint('loadRemote: failed to open IndexedDB');
        cbCancel();
    };

    rq.onblocked = function (event) {
        cbPrint('loadRemote: failed to open IndexedDB: blocked');
        cbCancel();
    };

    rq.onabort = function (event) {
        cbPrint('loadRemote: failed to open IndexedDB: abort');
        cbCancel();
    };
}
