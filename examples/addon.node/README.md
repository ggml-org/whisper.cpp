# addon

This is an addon demo that can **perform whisper model reasoning in `node` and `electron` environments**, based on [cmake-js](https://github.com/cmake-js/cmake-js).
It can be used as a reference for using the whisper.cpp project in other node projects.

## Install

```shell
npm install
```

## Compile

Make sure it is in the project root directory and compiled with make-js.

```shell
npx cmake-js compile -T addon.node -B Release
```

For Electron addon and cmake-js options, you can see [cmake-js](https://github.com/cmake-js/cmake-js) and make very few configuration changes.

> Such as appointing special cmake path:
> ```shell
> npx cmake-js compile -c 'xxx/cmake' -T addon.node -B Release
> ```

## Run

```shell
cd examples/addon.node

node index.js --language='language' --model='model-path' --fname_inp='file-path'
```

Because this is a simple Demo, only the above parameters are set in the node environment.

Other parameters can also be specified in the node environment.
