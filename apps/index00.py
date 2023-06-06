from dash import Dash, html
from components.layout import create_layout

def main() -> None:
    app = Dash()
    app.title = 'Number Classifier'
    app.layout = create_layout
    app.run()

if __name__ == '__main__':
    main()
