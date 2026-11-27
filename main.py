""" Module main: Main module to drive everything """

from monte_carlo import MonteCarlo

def main() -> None:
    """Main function to drive everything"""

    mc = MonteCarlo()
    mc.run()

if __name__ == "__main__":
    main()
