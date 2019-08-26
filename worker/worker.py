from time import sleep


def run():
    while True:
        print('Worker, sleeping for 1 minute')
        sleep(60)
        continue


if __name__ == '__main__':
    run()
