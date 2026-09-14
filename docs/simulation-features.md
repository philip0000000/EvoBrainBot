# Simulation features

This is the inventory of behaviorally relevant world and agent features, not a
list of console options, rendering features, or performance tooling.

**Keep this document current:** whenever a simulation ability, trait, sensor,
resource, terrain rule, lifecycle rule, or generation default changes, update the
affected rows and supporting tables in the same change. Describe implemented
behavior, not planned behavior.

Values below are current default presets. Explicit configuration and saved
settings can differ. “How common” means a fixed count, a generation probability,
or which agents possess the feature—not a prediction of evolved populations.
Random generation is seed-deterministic; a probability is not an exact map quota.

## World features

| Feature | Behavior | How common | Generation / calculation |
| --- | --- | --- | --- |
| World size and wrapping | Continuous positions in a toroidal world; opposite edges connect. | Every world. Small is default. | Small 5×5; medium 10×10; large 20×20. Area multipliers are 1×/4×/16×. |
| Simulation time and daylight | One smooth global day/night cycle changes sight range, not terrain or food color seen by agents. | Entire world; one cycle per 2,000 completed ticks. | Phase = (tick modulo cycle length) / cycle length; light = 0.5 − 0.5 × cos(2π × phase). Sight = 0.10 + 0.15 × light. No independent mutable clock state. |
| Islands and ocean | Rounded islands share surrounding water; connections wrap across map edges. | Exactly 1 / 2 / 3 main islands on small / medium / large. | Seeded center templates are translated and optionally transposed. Normalized base radii 0.34 / 0.21 / 0.19 receive independent axis factors 0.94–1.04, random rotation and sinusoidal coastal bulges. |
| Major wetlands | Island-sized rounded patchwork landmasses beside and overlapping parent islands; existing mainland stays solid. | Exactly 0 / 1 / 2 on small / medium / large. | Different parent islands are selected in cyclic order from a random starting parent. Each wetland uses 95% of its parent's ellipse radii. A 64-direction search chooses the most open placement relative to other islands/wetlands; center spacing is 1.65 × the parent's smaller radius. |
| Wetland land/water mixture | Three-cell-wide patches can be land or water; no coastal wetness gradient. Individual land fragments need not connect. | Each wetland has its own water probability, uniformly sampled from 0.25–0.75. | A seed/source/patch-coordinate hash is compared with that probability. Every offshore patch uses the same probability within its wetland. Rounded boundaries and solid mainland can clip a patch. Actual water percentage need not equal the sampled probability exactly. |
| Fertility regions | Rectangular regions tile the world without gaps or overlaps, independently of coastline and medium. | Cover the whole map. | Axis partitions reserve space for remaining regions and use bounded random widths/heights, normally 0.50–1.25 world units. Fertility rectangles do not clip islands or wetlands. A wetland region marker identifies its source/attachment, not its full footprint. |
| Ordinary fertility | Controls food supply, not food type. Both water and land can be fertile. | Every region not selected as sparse. | Uniform fertility 0.50–1.00, independent of land/water. |
| Sparse/desert fertility | Very low but nonzero food supply on either medium. | Small: 10% chance of exactly one region. Medium: 20% chance of exactly one. Otherwise none. Large: approximately 15% of regions. | Selected regions get uniform fertility 0.05–0.15. Large selects round(region count × 0.15) regions through neighboring growth, in batches of up to eight; adjacency wraps and clusters can meet. |
| Rock cells | One impassable square obstacle type; blocks movement and vision. | Scattered candidate chance 1.2% per terrain cell, plus cave walls. | Terrain grid targets 0.04-unit cells. Stable coordinate hashes select scattered rocks. These are candidate probabilities: entrance clearing and the guaranteed central spawn clearing remove some rocks. |
| Caves | Connected rock-wall formations with passages and openings; same rock type as scattered rocks. | Cave layout selected independently per fertility region with probability 1/7. | A deterministic region hash selects cave regions. Local columns repeat walls every seven cells and rows every nine, with periodic gaps. This is not a fixed number of caves per world. |
| Cave entrances | Openings remain free of crossing walls and scattered rocks. | Each entrance: 70% two cells wide, 30% three cells wide. | A stable hash of seed, region, wall orientation/index, and gap selects the width. All cells belonging to the opening share that selection. |
| Plant food placement | One plant-food type, available in land and water, outside rocks. | Initial/bootstrap targets: 4,000 / 16,000 / 64,000 items. Later targets depend on population; see table below. | Bootstrap ignores fertility and places food across traversable terrain. Outside bootstrap, fertility weights placement. Food counts and per-tick addition limits scale by area, but population thresholds scale 1×/2×/4×. |
| Food energy and regrowth | Food is gradually consumed and surviving items regrow. | Every surviving food item receives a pulse each 100 completed ticks. | New food starts at 0.25 energy. A global pulse adds 0.025, clamped at 0.25. Empty consumed items are removed. |
| Population support and food limits | Random founders restore the population floor; food supply responds to population bands. There is no hard reproduction/population cap. | Starting population and floor: 30 / 60 / 120. | After deaths, founders are added until the floor is restored. New food stops at 500 / 1,000 / 2,000 agents and resumes below that threshold. Existing food is not deleted when a band changes and still regrows. |
| Reproducibility and persistence | Seed/configuration regenerate terrain; checkpoints preserve agents, tick and explicit simulation settings. | Every world; checkpoint format version 16. | Terrain generation has its own RNG, independent of the tick RNG. Loading does not reapply world-size scaling. Format changes reject incompatible old saves rather than silently regenerate different terrain. |

## Agent features

For agents, initialization describes random founders and inherited offspring
separately. Trait frequencies after evolution depend on selection; they are not
fixed world-generation percentages.

| Feature | Behavior | How common | Initialization / calculation |
| --- | --- | --- | --- |
| Founder placement and body | Continuous position/direction; fixed radius 0.010. | All initial and population-floor agents. | Random non-rock cells and positions are retried until the body clears rocks. Direction is uniform over a full turn. Initial energy is 0.5. |
| Vision | Two eyes, three first-hit rays per eye; each supplies RGB and proximity for agents, food or rocks. | All agents: six rays, 24 vision inputs. | Daylight interpolates range from 0.10 to 0.25. Nearer objects occlude farther ones. Rocks provide grey RGB (0.58, 0.58, 0.58). Ground color/fertility and medium are not directly visible. |
| Internal sensors | Normalized energy, previous bite damage, oxygen reserve and rock contact. | All agents: four additional brain inputs. | State is supplied to the next brain evaluation. There is no dedicated daylight or land/water input and no agent-contact sensor. |
| Brain and actions | 28 inputs, capacity for 16 hidden nodes, three outputs: turn, move and eat. Recurrent connections are possible. | All founders start with eight active hidden nodes; descendants can activate the other eight. | Founder parameters are randomized from −1 to +1 in the starting topology. Mutation can change parameters/topology; recurrent memory resets for newborns. Brain parameter bounds are −4 to +4. See [brain architecture](brain-architecture.md). |
| Movement and turning | Brain-controlled continuous movement; rocks block movement. | All agents. | Base maximum movement 0.01 units/tick and turn 0.25 radians/tick. Compatibility c is adaptation in water, 1 − adaptation on land. Speed multiplier = 0.5 + 0.5 × clamp(2c − 1, 0, 1). Perfect amphibians have 50% speed in both media; specialists have 100% in their preferred medium, 50% in the other. |
| Water adaptation / lungs | One inherited value controls both breathing and movement. | Every founder: 0.5 (amphibious). Descendants: 0, 0.25, 0.5, 0.75 or 1. | 0 is a land specialist; 1 is a water specialist. On a trait-mutation trigger, choose uniformly among all five values, including the parent's value. Founders are amphibious to support establishment in the food-rich bootstrap world. |
| Oxygen reserve | Fixed capacity; incompatibility drains oxygen and can cause energy loss. | All agents; newborns begin full. | Maximum 1.0. Per tick: clamp(oxygen + 0.012c − 0.008(1 − c), 0, 1). If the resulting reserve is zero, lose 0.005 energy. Capacity is not evolvable. |
| Diet / carnivore tendency | Plant efficiency = 1 − tendency; meat efficiency = tendency. | Every founder: 0 (herbivore). Descendants: 0, 0.25, 0.5, 0.75 or 1. | On a trait trigger, uniformly reselect one of the five values, including the current value. No periodic forced carnivore introductions. Full-bite energy outcomes are tabulated below. |
| Eating and biting | Agents can bite food or other agents; targets lose the bitten energy before digestion efficiency applies. | All agents have an eat output, regardless of diet. | Eat output ≥ 0.50 attempts eating, costing 0.001 even if unsuccessful. A bite transfers at most 0.05 target energy before diet efficiency; partial targets give less. |
| Body color | Normalized RGB, visible to other agents. | Every founder has independently randomized RGB channels in [0, 1). | Each channel has an independent trait-rate check per birth. A trigger adds exactly −0.15 or +0.15 with a 50/50 choice; reflect at 0 and 1 to keep values bounded. |
| Rock contact | Reports movement blocked by rocks without requiring damage. | Every agent; active only when rock contact occurs. | Movement sets contact state for the following brain tick. Physical contact with another agent is not reported by this sensor. |
| Energy and death | Living, movement, eating and suffocation spend energy. Death occurs at zero energy. | All agents, each tick. | Living cost 0.001/tick; movement cost 0.1 × movement distance; eating and suffocation costs as above. There is no brain-computation energy charge. |
| Reproduction | Eligible parents split energy with one child; traits and brain are inherited with mutation. | Whenever a surviving parent reaches the reproduction threshold, not a random world-generation chance. | Threshold is 1.0 energy. Newborn recurrent memory resets and oxygen starts full. Offspring are produced by reproduction, not by selecting a fresh random founder. |
| Brain mutation rate and strength | Separate inherited brain mutation controls; unchanged by trait-rate rules. | Every founder and descendant. | Founder rate uniform from 1/75 to 0.05; strength from 0.05 to 0.20. Inherited minimum for each is 1/75 and maximum is 1. Rate is a per-value probability; strength is a change scale, not a probability. |
| Trait mutation rate | Shared inherited trigger probability for lungs, diet and each RGB channel; each check is independent. | Allowed rates: 20%, 25%, …, 100%. Each founder rate has probability 1/17. | Birth checks use the parent's rate. Every child inherits a fair −5/+5 percentage-point adjustment, clamped to 20%–100%; its new rate governs its future offspring. A 100% trigger does not guarantee a different diet/lung value, since reselection may return the same value. |

## Diet energy per bite

For a full bite of `0.05` energy, each eating attempt costs `0.001`:

| Diet value | Plant energy gained | Meat energy gained | Eating cost | Net plant gain | Net meat gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0500` | `0.0000` | `0.001` | `+0.0490` | `-0.0010` |
| `0.25` | `0.0375` | `0.0125` | `0.001` | `+0.0365` | `+0.0115` |
| `0.5` | `0.0250` | `0.0250` | `0.001` | `+0.0240` | `+0.0240` |
| `0.75` | `0.0125` | `0.0375` | `0.001` | `+0.0115` | `+0.0365` |
| `1` | `0.0000` | `0.0500` | `0.001` | `-0.0010` | `+0.0490` |

Net gains exclude living and movement costs. Unsuccessful eating attempts still
cost energy, and partially depleted targets provide smaller bites. The target
loses the full bitten amount before the eater's digestion efficiency is applied.

## Plant-food recovery

Size presets preserve food density. Population settings scale by 2× for medium
and 4× for large, keeping population growth more conservative than area scaling:

| Setting | Small | Medium | Large |
| --- | ---: | ---: | ---: |
| Dimensions | 5×5 | 10×10 | 20×20 |
| Initial population / population floor | 30 | 60 | 120 |
| Bootstrap food target | 4,000 | 16,000 | 64,000 |
| Bootstrap below population | 50 | 100 | 200 |
| Boosted food target | 1,000 | 4,000 | 16,000 |
| Normal food target | 500 | 2,000 | 8,000 |
| Switch to normal food at population | 200 | 400 | 800 |
| Stop food spawning at population | 500 | 1,000 | 2,000 |
| Maximum new food per tick | 5 | 20 | 80 |

Stopping food spawning does not cap reproduction. Agent size, speed, sight,
day/night duration, food energy, and regrowth per item are identical across sizes.
Regions keep comparable dimensions; multiple fertility regions form each continent
or ocean. Larger worlds take longer to traverse and require more simulation work.

New small simulations place 30 herbivore founders and 4,000 full-energy plant-food
items in a 5 by 5 toroidal world. Below 50 living agents, food is placed uniformly
across traversable land and water cells so a random founder population can
bootstrap. From 50 through 199 agents, fertility controls new food placement and
the target is 1,000 items. From 200 through 499 agents, the target is 500 items.
At 500 agents or more, no new food is spawned. At most five whole food items are
added per completed tick, and crossing into a higher population band never
deletes excess food.

Every 100 completed ticks, one global regrowth pulse adds 0.025 energy to each
surviving food item, clamped to the configured maximum of 0.25. These initial
balance values are provisional and may change after smoke testing.

Ordinary population-floor founders begin with carnivore tendency zero. Diet then
changes only through inherited mutation and selection; the simulation does not
periodically inject carnivores.


## Implementation references

- [World/agent configuration and state](../include/evobrain/simulation.hpp)
- [Generation, sensing, lifecycle and mutation](../src/core/simulation.cpp)
- [Brain architecture](brain-architecture.md)
- [Checkpoint format](../src/core/checkpoint.cpp)
