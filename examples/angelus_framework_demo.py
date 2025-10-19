from angelus import AngelusFramework
from angelus.common.types import Intent


def main():
    framework = AngelusFramework()
    intent = Intent(user_id=None, text="Find clarity and protection for a decision", tags=["clarity", "protection"])
    guidance = framework.run(intent)
    print("Summary:", guidance.summary)
    print("Archetype:", guidance.archetype)
    print("Confidence:", guidance.confidence)
    for step in guidance.steps:
        print(" -", step)


if __name__ == "__main__":
    main()
