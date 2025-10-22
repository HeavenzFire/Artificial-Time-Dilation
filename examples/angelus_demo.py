from angelus import InvocationEngine, ResonanceOracle, LatticeWeaver
from angelus.common.types import Intent, SigilSpec
from angelus.common.sigil import generate_sigil_svg


def main():
    intent = Intent(user_id=None, text="Heal past trauma and find clarity", tags=["healing", "clarity"])

    inv = InvocationEngine()
    guidance = inv.invoke(intent)

    oracle = ResonanceOracle()
    refined = oracle.refine(guidance, intent)

    lattice = LatticeWeaver()
    lattice.add_intent(intent)
    lattice.connect_similarity(intent, Intent(None, "Seek clarity for decisions", tags=["clarity"]))

    svg = generate_sigil_svg(SigilSpec(seed=intent.text))

    print("Summary:", refined.summary)
    print("Archetype:", refined.archetype)
    print("Confidence:", refined.confidence)
    print("Steps:")
    for s in refined.steps:
        print(" -", s)
    print("Neighborhood:", lattice.neighborhood(intent))

    with open("angelus_sigil.svg", "w", encoding="utf-8") as f:
        f.write(svg)
    print("Sigil saved to angelus_sigil.svg")


if __name__ == "__main__":
    main()
