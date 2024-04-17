from src.models import App
from src.core.vo import vo_cicle
from config import setup_config
from src.viewer.vo_plots import create_plots


def main():
    app = App()
    setup_config(app)
    vo_cicle(app)
    # print(repr(app))
    create_plots(app)


if __name__ == "__main__":
    main()
