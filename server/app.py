from data_clean_env.server.app import app


def main():
    from data_clean_env.server.app import main as inner_main
    return inner_main()


if __name__ == "__main__":
    main()
