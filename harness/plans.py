"""Hand-written step plans for LIBERO-Long (M1-M5; an LLM writes them in M6).

One step per goal predicate of the task's BDDL goal, in the order of the
instruction; the predicate is the step's check. Predicates that already hold
at the start are not steps (task 8: the stove starts on). Task 5 has a
single predicate, so it stays one step (a control).

Two wordings per step:
  trained  the phrasing SmolVLA's training data uses for single-object tasks
           (LIBERO-Object: "pick up the X and place it in the basket";
           LIBERO-Goal: "turn on the stove", "put the bowl on the stove")
  split    the task's own sentence cut at "and"
They coincide where the trained phrasing is the split one.

Left/right follow LIBERO's convention: in task 4 "the left plate" is plate_1
at y = -0.32, so left is -y. In task 8 moka_pot_2 (y ~ 0.05) is the left pot
and moka_pot_1 (y ~ 0.25) the right one in every init state.

Hand-written plans use the task's goal from LIBERO's files, so runs with them
are upper bounds, not a fair comparison with the VLA (PLAN.md).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanStep:
    trained: str
    split: str
    check: tuple  # BDDL goal predicate, e.g. ("in", "alphabet_soup_1", "basket_1_contain_region")

    def instruction(self, wording):
        return {"trained": self.trained, "split": self.split}[wording]


def basket(obj, name, phrase):
    return PlanStep(f"pick up the {name} and place it in the basket",
                    f"put the {phrase} in the basket",
                    ("in", obj, "basket_1_contain_region"))


PLANS = {  # libero_10 task id -> steps
    0: [basket("alphabet_soup_1", "alphabet soup", "alphabet soup"),
        basket("tomato_sauce_1", "tomato sauce", "tomato sauce")],
    1: [basket("cream_cheese_1", "cream cheese", "cream cheese box"),
        basket("butter_1", "butter", "butter")],
    2: [PlanStep("turn on the stove", "turn on the stove", ("turnon", "flat_stove_1")),
        PlanStep("put the moka pot on the stove", "put the moka pot on it",
                 ("on", "moka_pot_1", "flat_stove_1_cook_region"))],
    3: [PlanStep("put the black bowl in the bottom drawer of the cabinet",
                 "put the black bowl in the bottom drawer of the cabinet",
                 ("in", "akita_black_bowl_1", "white_cabinet_1_bottom_region")),
        PlanStep("close the bottom drawer of the cabinet", "close it",
                 ("close", "white_cabinet_1_bottom_region"))],
    4: [PlanStep("put the white mug on the left plate", "put the white mug on the left plate",
                 ("on", "porcelain_mug_1", "plate_1")),
        PlanStep("put the yellow and white mug on the right plate",
                 "put the yellow and white mug on the right plate",
                 ("on", "white_yellow_mug_1", "plate_2"))],
    5: [PlanStep("pick up the book and place it in the back compartment of the caddy",
                 "pick up the book and place it in the back compartment of the caddy",
                 ("in", "black_book_1", "desk_caddy_1_back_contain_region"))],
    6: [PlanStep("put the white mug on the plate", "put the white mug on the plate",
                 ("on", "porcelain_mug_1", "plate_1")),
        PlanStep("put the chocolate pudding to the right of the plate",
                 "put the chocolate pudding to the right of the plate",
                 ("on", "chocolate_pudding_1", "living_room_table_plate_right_region"))],
    7: [basket("alphabet_soup_1", "alphabet soup", "alphabet soup"),
        basket("cream_cheese_1", "cream cheese", "cream cheese box")],
    8: [PlanStep("put the left moka pot on the stove", "put the left moka pot on the stove",
                 ("on", "moka_pot_2", "flat_stove_1_cook_region")),
        PlanStep("put the right moka pot on the stove", "put the right moka pot on the stove",
                 ("on", "moka_pot_1", "flat_stove_1_cook_region"))],
    9: [PlanStep("put the yellow and white mug in the microwave",
                 "put the yellow and white mug in the microwave",
                 ("in", "white_yellow_mug_1", "microwave_1_heating_region")),
        PlanStep("close the microwave", "close it", ("close", "microwave_1"))],
}
