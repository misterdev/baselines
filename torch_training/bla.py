import sys

print("bla")
def main(argv):
    print("main bla {}".format(argv))


if __name__ == '__main__':
    main(sys.argv[1:])
