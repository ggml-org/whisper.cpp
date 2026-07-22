# chessboard.js

chessboard.js is a JavaScript chessboard component. It depends on [jQuery].

Please see [chessboardjs.com] for documentation and examples.

## What is chessboard.js?

chessboard.js is a JavaScript chessboard component with a flexible "just a
board" API that

chessboard.js is a standalone JavaScript Chess Board. It is designed to be "just
a board" and expose a powerful API so that it can be used in different ways.
Here's a non-exhaustive list of things you can do with chessboard.js:

- Use chessboard.js to show game positions alongside your expert commentary.
- Use chessboard.js to have a tactics website where users have to guess the best
  move.
- Integrate chessboard.js and [chess.js] with a PGN database and allow people to
  search and playback games (see [Example 5000])
- Build a chess server and have users play their games out using the
  chessboard.js board.

chessboard.js is flexible enough to handle any of these situations with relative
ease.

## What can chessboard.js **not** do?

The scope of chessboard.js is limited to "just a board." This is intentional and
makes chessboard.js flexible for handling a multitude of chess-related problems.

This is a common source of confusion for new users. [remove?]

Specifically, chessboard.js does not understand anything about how the game of
chess is played: how a knight moves, who's turn is it, is White in check?, etc.

Fortunately, the powerful [chess.js] library deals with exactly this sort of
problem domain and plays nicely with chessboard.js's flexible API. Some examples
of chessboard.js combined with chess.js: 5000, 5001, 5002

Please see the powerful [chess.js] library for an API to deal with these sorts
of questions.


This logic is distinct from the logic of the board. Please see the powerful
[chess.js] library for this aspect of your application.



Here is a list of things that chessboard.js is **not**:

- A chess engine
- A legal move validator
- A PGN parser

chessboard.js is designed to work well with any of those things, but the idea
behind chessboard.js is that the logic that controls the board should be
independent of those other problems.

## Docs and Examples

- Docs - <http://chessboardjs.com/docs>
- Examples - <http://chessboardjs.com/examples>

## Developer Tools

```sh
# create a build in the build/ directory
npm run build

# re-build the website
npm run website
```

## License

[MIT License](LICENSE.md)

[jQuery]:https://jquery.com/
[chessboardjs.com]:http://chessboardjs.com
[chess.js]:https://github.com/jhlywa/chess.js
[Example 5000]:http://chessboardjs.com/examples#5000
