#画像ベースサロゲート向けビューア
cd ModelViewer
bokeh serve --show MyBoke.py --args --ModelName [path/to/model.pth] --dsp 1 --dvo 0 --dvoe 0 --pixelshuffle
