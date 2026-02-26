# GirlBot AI Stylist - Professional System Architecture

## Research Foundation

This system design is based on best practices from leading styling platforms and scientific research:

**Industry Leaders Studied:**
- [Stitch Fix](https://algorithms-tour.stitchfix.com/) - Human + AI hybrid, 90 data points, GPT-4 for preference summarization
- [Sephora Color IQ](https://digiday.com/marketing/color-iq-sephoras-shade-matching-skin-care-tool-boosts-brand-loyalty/) - 14M+ matches, Pantone partnership
- [Style DNA](https://apps.apple.com/us/app/style-dna-ai-color-analysis/id1358319821) - 12-season color analysis from selfies
- [Klodsy](https://klodsy.com/blog/best-ai-stylist-apps-2026-comparison/) - Wardrobe digitization, virtual try-on

**Scientific Frameworks:**
- [12-Season Color Analysis](https://radiantlydressed.com/explore-the-12-seasons-of-color/) - Professional draping methodology
- [Kibbe Body Type System](https://theconceptwardrobe.com/kibbe-body-types/an-introduction-to-the-kibbe-body-types) - 13 types based on yin/yang balance
- [Enclothed Cognition](https://pmc.ncbi.nlm.nih.gov/articles/PMC8455911/) - Psychology of dress and confidence
- [Capsule Wardrobe Methodology](https://capsule.style/en/blogs/capsule-blog-4/capsule-wardrobe-defined) - 15-25 piece intentional curation

---

## Core Design Principles

### 1. Human-Centered AI (Stitch Fix Model)
> "We find that the human touch helps us drive deeper relationships with our clients, and building those client relationships drives higher loyalty." - Noah Zamansky, Stitch Fix VP Product

**Implementation:**
- AI provides analysis and recommendations
- Always explain the "why" behind suggestions
- Enable user feedback loops to refine recommendations
- Conversational, not robotic - feels like texting a stylish friend

### 2. Progressive Profiling
> Stitch Fix collects 90 data points upfront; we collect progressively through natural conversation.

**Implementation:**
- Start with minimal data (just a selfie)
- Build profile organically through interactions
- Never interrogate - learn through helpfulness
- Profile confidence score increases over time

### 3. Science-Backed Personalization
> "Color analysis is not guesswork—it's a learned skill backed by theory, practice, and science." - Style Academy International

**Implementation:**
- Use established color theory (12-season system)
- Apply body type science (Kibbe/classic systems)
- Reference fashion psychology research
- Provide reasoning users can understand

### 4. Body Positivity
> "The Kibbe system celebrates diversity and helps each person enhance their natural features, rather than comparing types hierarchically."

**Implementation:**
- No body type is "better" or "worse"
- Focus on harmony, not fixing "flaws"
- Celebrate individual characteristics
- Language is always empowering, never critical

---

## System Components

### Component 1: Style DNA Profile

**Data Model - Comprehensive User Profile:**

```python
@dataclass
class StyleDNA:
    """Complete style identity - the user's 'Style DNA'"""

    # === TIER 1: ESSENTIAL (Collected First) ===
    telegram_id: int

    # Color Analysis (from selfie)
    skin_undertone: Literal["warm", "cool", "neutral", "olive"]
    color_season: str  # 12-season classification
    color_season_confidence: float  # 0-1
    best_colors: list[ColorSwatch]      # With hex, name, category
    worst_colors: list[ColorSwatch]     # Colors to avoid
    neutral_palette: list[ColorSwatch]  # Personal neutrals
    metal_tone: Literal["gold", "silver", "rose_gold", "mixed"]

    # === TIER 2: BODY & FIT (Collected Second) ===

    # Body Geometry (from full-body photo or quiz)
    body_type_classic: str  # hourglass, pear, apple, rectangle, inverted_triangle
    body_type_kibbe: str | None  # For users wanting deeper analysis
    height_category: Literal["petite", "average", "tall"]  # <5'3", 5'3"-5'7", >5'7"
    vertical_line: Literal["short", "moderate", "long"]  # Torso vs leg ratio

    # Fit Preferences (learned)
    preferred_fit_tops: Literal["fitted", "relaxed", "oversized"]
    preferred_fit_bottoms: Literal["fitted", "relaxed", "wide"]
    comfort_zones: list[str]  # Body parts they like to show
    coverage_preferences: list[str]  # Areas they prefer covered

    # === TIER 3: STYLE IDENTITY (Built Over Time) ===

    # Style Archetype
    primary_archetype: str  # classic, romantic, dramatic, natural, gamine, ethereal
    secondary_archetype: str | None
    aesthetic_keywords: list[str]  # "minimalist", "boho", "edgy", "preppy"
    style_icons: list[str]  # People whose style they admire

    # === TIER 4: LIFESTYLE CONTEXT ===

    # Life Circumstances
    lifestyle_breakdown: dict[str, float]  # {"work": 0.5, "casual": 0.3, "events": 0.2}
    work_dress_code: str  # formal, business_casual, smart_casual, creative, none
    climate_zone: str  # tropical, temperate, cold, variable
    typical_activities: list[str]  # ["office", "gym", "dating", "mom_duties"]

    # === TIER 5: PRACTICAL CONSTRAINTS ===

    # Budget & Shopping
    budget_tier: Literal["budget", "mid_range", "premium", "luxury"]
    price_sensitivity: dict[str, tuple]  # {"tops": (20, 80), "shoes": (50, 200)}
    preferred_brands: list[str]
    avoided_brands: list[str]
    shopping_preference: Literal["online", "in_store", "mixed"]

    # Fabric & Material
    loved_fabrics: list[str]  # ["cotton", "silk", "cashmere"]
    avoided_fabrics: list[str]  # ["polyester", "wool_allergy"]
    texture_preference: Literal["smooth", "textured", "mixed"]

    # === TIER 6: MAKEUP PROFILE ===

    makeup_comfort: Literal["none", "minimal", "everyday", "full_glam"]
    foundation_undertone: Literal["warm", "cool", "neutral"]
    best_lip_colors: list[str]
    best_eye_colors: list[str]
    best_blush_tones: list[str]
    skin_concerns: list[str]  # For product recommendations

    # === LEARNING & FEEDBACK ===

    outfit_feedback: list[OutfitFeedback]  # Liked/disliked with reasons
    style_preferences_learned: dict  # Extracted patterns
    recommendation_accuracy: float  # Track how often they like our suggestions

    # === METADATA ===

    profile_completeness: float  # 0-1, how much we know
    created_at: datetime
    last_interaction: datetime
    total_interactions: int


@dataclass
class ColorSwatch:
    """A specific color with metadata"""
    hex: str
    name: str
    category: str  # "accent", "neutral", "avoid"
    wear_for: list[str]  # ["statement piece", "accessory", "avoid near face"]


@dataclass
class OutfitFeedback:
    """Feedback on a specific outfit recommendation or analysis"""
    outfit_id: str
    rating: Literal["loved", "liked", "neutral", "disliked"]
    feedback_text: str | None
    specific_likes: list[str]  # ["the colors", "the silhouette"]
    specific_dislikes: list[str]  # ["too tight", "wrong occasion"]
    would_wear_for: list[str]  # Occasions they'd actually wear it
    timestamp: datetime
```

---

### Component 2: Color Analysis Engine

**12-Season System Implementation:**

```python
COLOR_SEASONS = {
    # === SPRING (Warm + Bright/Light) ===
    "light_spring": {
        "characteristics": {
            "undertone": "warm",
            "value": "light",
            "chroma": "clear",
            "contrast": "low_to_medium",
            "description": "Delicate warmth, like early spring sunshine"
        },
        "typical_features": {
            "skin": ["ivory", "peachy", "light golden"],
            "hair": ["golden blonde", "strawberry blonde", "light auburn"],
            "eyes": ["light blue", "green", "light hazel"]
        },
        "best_colors": [
            {"hex": "#FFDAB9", "name": "Peach Puff", "use": "near face"},
            {"hex": "#98D8C8", "name": "Aqua Mint", "use": "accent"},
            {"hex": "#F0E68C", "name": "Light Gold", "use": "accent"},
            {"hex": "#E6E6FA", "name": "Lavender", "use": "accent"},
            {"hex": "#FFF5EE", "name": "Seashell", "use": "neutral"},
        ],
        "neutral_palette": [
            {"hex": "#FFFAF0", "name": "Floral White"},
            {"hex": "#F5DEB3", "name": "Warm Beige"},
            {"hex": "#D2B48C", "name": "Tan"},
            {"hex": "#DEB887", "name": "Camel"},
        ],
        "avoid_colors": [
            {"hex": "#000000", "name": "Black", "reason": "Too harsh, overwhelms delicate coloring"},
            {"hex": "#800000", "name": "Burgundy", "reason": "Too heavy and muted"},
            {"hex": "#808080", "name": "Gray", "reason": "Drains warmth from skin"},
        ],
        "metals": ["light_gold", "rose_gold"],
        "makeup": {
            "foundation": "Light with peachy/golden undertone",
            "lips": ["peachy pink", "light coral", "warm nude"],
            "eyes": ["champagne", "soft peach", "light brown"],
            "blush": ["peach", "light coral"],
            "avoid": ["anything too dark or cool-toned"]
        }
    },

    "true_spring": {
        "characteristics": {
            "undertone": "warm",
            "value": "medium_light",
            "chroma": "bright_clear",
            "contrast": "medium",
            "description": "Warm, bright, and clear like a sunny spring day"
        },
        "typical_features": {
            "skin": ["golden", "peachy", "warm beige"],
            "hair": ["golden brown", "auburn", "warm medium brown"],
            "eyes": ["warm green", "hazel", "golden brown", "turquoise"]
        },
        "best_colors": [
            {"hex": "#FF6B35", "name": "Coral", "use": "power color"},
            {"hex": "#4ECDC4", "name": "Turquoise", "use": "accent"},
            {"hex": "#F7DC6F", "name": "Sunny Yellow", "use": "accent"},
            {"hex": "#52BE80", "name": "Spring Green", "use": "accent"},
            {"hex": "#F1948A", "name": "Salmon", "use": "near face"},
        ],
        "neutral_palette": [
            {"hex": "#FFFFF0", "name": "Ivory"},
            {"hex": "#D2B48C", "name": "Camel"},
            {"hex": "#8B7355", "name": "Warm Brown"},
            {"hex": "#F5F5DC", "name": "Beige"},
        ],
        "avoid_colors": [
            {"hex": "#000000", "name": "Black", "reason": "Opt for chocolate brown instead"},
            {"hex": "#4A4A4A", "name": "Charcoal", "reason": "Too cool and heavy"},
            {"hex": "#800020", "name": "Burgundy", "reason": "Too muted for your brightness"},
        ],
        "metals": ["gold", "brass", "rose_gold"],
        "makeup": {
            "foundation": "Warm golden or peachy undertone",
            "lips": ["coral", "peach", "warm pink", "poppy red"],
            "eyes": ["warm browns", "peach", "gold", "turquoise"],
            "blush": ["peach", "coral", "warm pink"],
            "avoid": ["cool pink", "berry tones", "gray"]
        }
    },

    "clear_spring": {
        "characteristics": {
            "undertone": "warm_leaning_neutral",
            "value": "medium",
            "chroma": "high_bright",
            "contrast": "high",
            "description": "High contrast and brightness, warm but clear"
        },
        # ... continue for all 12 seasons
    },

    # === SUMMER (Cool + Muted/Soft) ===
    "light_summer": { ... },
    "true_summer": { ... },
    "soft_summer": { ... },

    # === AUTUMN (Warm + Muted/Deep) ===
    "soft_autumn": { ... },
    "true_autumn": {
        "characteristics": {
            "undertone": "warm",
            "value": "medium_to_deep",
            "chroma": "muted_rich",
            "contrast": "medium",
            "description": "Rich, warm, earthy - like autumn leaves"
        },
        "typical_features": {
            "skin": ["golden beige", "warm olive", "bronze"],
            "hair": ["auburn", "chestnut", "warm brown", "copper"],
            "eyes": ["hazel", "warm brown", "olive green", "amber"]
        },
        "best_colors": [
            {"hex": "#CC5500", "name": "Burnt Orange", "use": "power color"},
            {"hex": "#808000", "name": "Olive", "use": "neutral/accent"},
            {"hex": "#DAA520", "name": "Mustard", "use": "accent"},
            {"hex": "#008080", "name": "Teal", "use": "accent"},
            {"hex": "#A0522D", "name": "Sienna", "use": "neutral"},
            {"hex": "#CD853F", "name": "Peru/Caramel", "use": "near face"},
        ],
        "neutral_palette": [
            {"hex": "#F5F5DC", "name": "Cream"},
            {"hex": "#D2B48C", "name": "Camel"},
            {"hex": "#8B4513", "name": "Chocolate Brown"},
            {"hex": "#556B2F", "name": "Dark Olive"},
        ],
        "avoid_colors": [
            {"hex": "#000000", "name": "Black", "reason": "Too stark - use chocolate or charcoal brown"},
            {"hex": "#FF69B4", "name": "Hot Pink", "reason": "Too cool and bright"},
            {"hex": "#00BFFF", "name": "Electric Blue", "reason": "Too cool for your warmth"},
        ],
        "metals": ["gold", "brass", "copper", "bronze"],
        "makeup": {
            "foundation": "Warm with golden/olive undertone",
            "lips": ["terracotta", "brick red", "warm nude", "rust"],
            "eyes": ["copper", "bronze", "olive", "warm brown", "teal"],
            "blush": ["terracotta", "warm peach", "brick"],
            "avoid": ["cool pink", "frosty shades", "silver"]
        }
    },
    "deep_autumn": { ... },

    # === WINTER (Cool + Bright/Deep) ===
    "cool_winter": { ... },
    "true_winter": {
        "characteristics": {
            "undertone": "cool",
            "value": "ranges",
            "chroma": "bright_clear",
            "contrast": "high",
            "description": "High contrast, cool and clear like a crisp winter day"
        },
        "typical_features": {
            "skin": ["porcelain", "olive with cool undertone", "deep brown with cool undertone"],
            "hair": ["black", "dark brown", "silver gray"],
            "eyes": ["dark brown", "black-brown", "cool blue", "gray"]
        },
        "best_colors": [
            {"hex": "#FF0000", "name": "True Red", "use": "power color"},
            {"hex": "#0000CD", "name": "Royal Blue", "use": "power color"},
            {"hex": "#FF1493", "name": "Hot Pink", "use": "accent"},
            {"hex": "#00CED1", "name": "Dark Turquoise", "use": "accent"},
            {"hex": "#9400D3", "name": "Violet", "use": "accent"},
            {"hex": "#FFFFFF", "name": "Pure White", "use": "neutral"},
            {"hex": "#000000", "name": "True Black", "use": "neutral"},
        ],
        "neutral_palette": [
            {"hex": "#FFFFFF", "name": "Pure White"},
            {"hex": "#000000", "name": "Black"},
            {"hex": "#36454F", "name": "Charcoal"},
            {"hex": "#000080", "name": "Navy"},
        ],
        "avoid_colors": [
            {"hex": "#FFA500", "name": "Orange", "reason": "Wrong undertone entirely"},
            {"hex": "#F5DEB3", "name": "Beige/Wheat", "reason": "Too warm and muted"},
            {"hex": "#808000", "name": "Olive", "reason": "Muddies your clear coloring"},
        ],
        "metals": ["silver", "platinum", "white_gold"],
        "makeup": {
            "foundation": "Cool with pink undertone or cool olive",
            "lips": ["true red", "berry", "fuchsia", "wine", "cool plum"],
            "eyes": ["charcoal", "plum", "navy", "silver", "cool gray"],
            "blush": ["cool pink", "berry", "plum"],
            "avoid": ["orange", "peach", "warm browns"]
        }
    },
    "deep_winter": { ... },
}
```

**Color Analysis Process:**

```python
async def perform_color_analysis(selfie_bytes: bytes) -> ColorAnalysisResult:
    """
    Professional-grade color analysis from a selfie.

    Methodology based on professional draping techniques:
    1. Analyze skin undertone (warm/cool/neutral/olive)
    2. Determine value (light/medium/deep)
    3. Assess chroma (muted/bright/soft)
    4. Calculate contrast level
    5. Match to closest season
    """

    # GPT-4V analysis prompt (professional methodology)
    analysis_prompt = """
    You are a professional color analyst trained in the 12-season Sci/Art method.

    Analyze this person's NATURAL coloring to determine their color season.

    STEP 1 - UNDERTONE ANALYSIS:
    Examine the skin for warmth vs coolness:
    - Warm: Golden, peachy, yellowish undertones
    - Cool: Pink, rosy, bluish undertones
    - Neutral: Mix of both, or very balanced
    - Olive: Green-gray undertone (can be warm or cool olive)

    STEP 2 - VALUE ANALYSIS:
    How light or deep is their overall coloring?
    - Light: Fair skin, light hair, light eyes
    - Medium: Medium skin, medium hair
    - Deep: Dark skin, dark hair, dark eyes

    STEP 3 - CHROMA ANALYSIS:
    How muted or clear/bright is their coloring?
    - Muted/Soft: Colors blend together, low saturation appearance
    - Bright/Clear: Colors are distinct, high saturation appearance

    STEP 4 - CONTRAST ANALYSIS:
    What's the difference between their lightest and darkest features?
    - Low: Features similar in value
    - Medium: Moderate difference
    - High: Stark difference (e.g., dark hair + light skin)

    STEP 5 - SEASON DETERMINATION:
    Based on the above, classify into one of 12 seasons:

    SPRING (warm undertone):
    - Light Spring: warm + light + somewhat muted
    - True Spring: warm + medium + bright/clear
    - Clear Spring: warm-neutral + high contrast + very bright

    SUMMER (cool undertone):
    - Light Summer: cool + light + soft
    - True Summer: cool + medium + muted
    - Soft Summer: cool-neutral + low contrast + very muted

    AUTUMN (warm undertone):
    - Soft Autumn: warm-neutral + low contrast + muted
    - True Autumn: warm + medium + rich/muted
    - Deep Autumn: warm + deep + rich

    WINTER (cool undertone):
    - Cool Winter: cool + medium + bright
    - True Winter: cool + high contrast + very bright
    - Deep Winter: cool + very deep + rich

    Provide your analysis with reasoning for each step.
    """

    # ... vision API call and processing
```

---

### Component 3: Body Type Analysis

**Dual System Approach:**

```python
# Classic 5-Type System (Simple, Widely Understood)
CLASSIC_BODY_TYPES = {
    "hourglass": {
        "characteristics": "Bust and hips roughly equal, defined waist",
        "strengths": ["balanced proportions", "natural curves"],
        "styling_goals": ["highlight waist", "balanced looks"],
        "best_silhouettes": ["fitted waist", "wrap styles", "belted looks"],
        "recommended": {
            "tops": ["wrap tops", "v-necks", "fitted blouses"],
            "bottoms": ["mid-rise", "a-line skirts", "straight leg"],
            "dresses": ["wrap dresses", "fit and flare", "bodycon"]
        }
    },
    "pear": {
        "characteristics": "Hips wider than shoulders/bust",
        "strengths": ["feminine curves", "graceful lower half"],
        "styling_goals": ["balance top and bottom", "draw eye up"],
        "best_silhouettes": ["a-line", "wide necklines", "structured shoulders"],
        "recommended": {
            "tops": ["boat necks", "off-shoulder", "statement necklines"],
            "bottoms": ["a-line skirts", "bootcut", "wide leg"],
            "dresses": ["fit and flare", "empire waist"]
        }
    },
    "apple": {
        "characteristics": "Midsection fuller, often with great legs",
        "strengths": ["great legs", "often good bust"],
        "styling_goals": ["elongate torso", "show off legs"],
        "best_silhouettes": ["empire waist", "v-necks", "straight lines"],
        "recommended": {
            "tops": ["v-necks", "empire waist", "tunics"],
            "bottoms": ["straight leg", "bootcut", "mid-rise"],
            "dresses": ["empire waist", "wrap", "shift"]
        }
    },
    "rectangle": {
        "characteristics": "Shoulders, waist, hips similar width",
        "strengths": ["balanced frame", "athletic look", "can wear anything"],
        "styling_goals": ["create curves", "define waist"],
        "best_silhouettes": ["peplum", "ruffles", "layered looks"],
        "recommended": {
            "tops": ["peplum", "ruffled", "cropped jackets"],
            "bottoms": ["high-waisted", "flared", "pleated"],
            "dresses": ["fit and flare", "belted", "bodycon"]
        }
    },
    "inverted_triangle": {
        "characteristics": "Shoulders wider than hips",
        "strengths": ["strong shoulders", "athletic upper body"],
        "styling_goals": ["balance with lower half", "soften shoulders"],
        "best_silhouettes": ["a-line", "wide leg", "full skirts"],
        "recommended": {
            "tops": ["v-necks", "scoop necks", "raglan sleeves"],
            "bottoms": ["a-line", "wide leg", "full skirts"],
            "dresses": ["fit and flare", "drop waist"]
        }
    }
}

# Kibbe System (Advanced, Optional)
KIBBE_TYPES = {
    "dramatic": {
        "yin_yang": "extreme yang",
        "essence": "Bold, sharp, striking",
        "bone_structure": "Long, angular, narrow",
        "best_lines": ["long vertical", "sharp angles", "sleek"],
        "avoid": ["fussy details", "rounded shapes", "small prints"]
    },
    "soft_dramatic": {
        "yin_yang": "yang dominant with yin undercurrent",
        "essence": "Bold glamour with softness",
        "bone_structure": "Long with soft flesh",
        "best_lines": ["vertical with drape", "bold but soft"],
        "avoid": ["stiff fabrics", "small details"]
    },
    # ... remaining Kibbe types
}
```

---

### Component 4: Wardrobe Intelligence

**Digital Wardrobe System:**

```python
@dataclass
class WardrobeItem:
    """A cataloged wardrobe item with full metadata"""

    id: str
    user_id: int

    # Classification
    category: str  # "tops", "bottoms", "dresses", etc.
    subcategory: str  # "blouse", "jeans", "midi_dress"

    # Visual Properties
    primary_color: ColorSwatch
    secondary_colors: list[ColorSwatch]
    pattern: str  # "solid", "stripes", "floral", "plaid", "abstract"
    pattern_scale: str | None  # "small", "medium", "large"

    # Garment Properties
    fabric: str  # "cotton", "silk", "polyester blend"
    weight: str  # "lightweight", "midweight", "heavy"
    structure: str  # "structured", "unstructured", "flowy"
    fit: str  # "fitted", "relaxed", "oversized"

    # Style Classification
    formality: int  # 1-10 scale
    style_tags: list[str]  # ["classic", "minimalist", "romantic"]
    season_appropriate: list[str]  # ["spring", "summer"]

    # Occasion Mapping
    occasions: list[str]  # ["work", "casual", "date", "formal"]

    # Usage Tracking
    times_worn: int
    last_worn: datetime | None
    times_recommended: int
    times_chosen_when_recommended: int

    # Outfit Compatibility (computed)
    pairs_well_with: list[str]  # IDs of compatible items
    avoid_with: list[str]  # IDs that clash

    # User Data
    purchase_date: datetime | None
    purchase_price: float | None
    brand: str | None
    notes: str | None

    # Media
    image_thumbnail: str  # Base64 encoded small image


class WardrobeAnalyzer:
    """Analyzes wardrobe for gaps, capsule building, and recommendations"""

    async def analyze_wardrobe(self, items: list[WardrobeItem], style_profile: StyleDNA) -> WardrobeAnalysis:
        """
        Comprehensive wardrobe analysis.

        Based on capsule wardrobe methodology:
        - Identify gaps in essentials
        - Assess color palette cohesion
        - Calculate versatility score
        - Suggest strategic additions
        """

        return WardrobeAnalysis(
            total_items=len(items),
            category_breakdown=self._count_by_category(items),
            color_palette=self._extract_color_palette(items),
            color_cohesion_score=self._calculate_color_cohesion(items, style_profile),
            style_consistency_score=self._calculate_style_consistency(items),
            versatility_score=self._calculate_versatility(items),
            outfit_combinations=self._count_possible_outfits(items),
            gaps=self._identify_gaps(items, style_profile),
            underutilized=self._find_underutilized(items),
            suggested_additions=self._suggest_additions(items, style_profile),
        )

    def _identify_gaps(self, items: list[WardrobeItem], profile: StyleDNA) -> list[WardrobeGap]:
        """
        Identify missing essentials based on lifestyle and capsule principles.

        Core capsule essentials:
        - Neutral base pieces (tops, bottoms)
        - Layering pieces
        - Occasion-appropriate items for their lifestyle
        """

        gaps = []

        # Check against lifestyle needs
        if profile.lifestyle_breakdown.get("work", 0) > 0.3:
            # They work - do they have work basics?
            work_items = [i for i in items if "work" in i.occasions]
            if len([i for i in work_items if i.category == "bottoms"]) < 2:
                gaps.append(WardrobeGap(
                    category="bottoms",
                    reason="Work wardrobe needs versatile bottoms",
                    suggestion=f"A {profile.neutral_palette[0].name.lower()} trouser",
                    priority="high"
                ))

        # Check color balance
        item_colors = [i.primary_color.hex for i in items]
        if not any(c in profile.neutral_palette for c in item_colors):
            gaps.append(WardrobeGap(
                category="any",
                reason="Missing items in your personal neutral colors",
                suggestion=f"Basics in {profile.neutral_palette[0].name}",
                priority="high"
            ))

        return gaps
```

---

### Component 5: Outfit Generation Engine

**Intelligent Outfit Assembly:**

```python
class OutfitEngine:
    """Generates outfit recommendations using wardrobe + style DNA"""

    def generate_outfit(
        self,
        occasion: str,
        wardrobe: list[WardrobeItem],
        style_profile: StyleDNA,
        weather: WeatherContext | None = None,
        mood: str | None = None,
    ) -> OutfitRecommendation:
        """
        Generate a complete outfit recommendation.

        Algorithm:
        1. Filter wardrobe by occasion appropriateness
        2. Apply color harmony rules (based on user's season)
        3. Apply body type flattery rules
        4. Consider weather/practical constraints
        5. Rank by user's historical preferences
        6. Return top recommendation with alternatives
        """

        # Step 1: Filter candidates
        candidates = self._filter_by_occasion(wardrobe, occasion)

        # Step 2: Apply color rules
        candidates = self._apply_color_harmony(candidates, style_profile)

        # Step 3: Build outfit combinations
        combinations = self._generate_combinations(candidates, style_profile)

        # Step 4: Score and rank
        scored = self._score_combinations(combinations, style_profile, occasion, mood)

        # Step 5: Return best with explanation
        best = scored[0]
        return OutfitRecommendation(
            items=best.items,
            score=best.score,
            reasoning=self._generate_reasoning(best, style_profile),
            alternatives=scored[1:4],
            styling_tips=self._generate_styling_tips(best, style_profile),
            shopping_suggestions=self._suggest_missing_pieces(best) if best.has_gaps else None,
        )

    def _apply_color_harmony(
        self,
        items: list[WardrobeItem],
        profile: StyleDNA
    ) -> list[WardrobeItem]:
        """
        Filter and score items by color harmony with user's season.

        Rules:
        - Items in user's best colors: +2
        - Items in user's neutral palette: +1
        - Items in user's avoid colors: -2 (or exclude near face)
        """

        for item in items:
            color_score = 0

            if item.primary_color.hex in [c.hex for c in profile.best_colors]:
                color_score += 2
                item.color_placement = "anywhere"
            elif item.primary_color.hex in [c.hex for c in profile.neutral_palette]:
                color_score += 1
                item.color_placement = "anywhere"
            elif item.primary_color.hex in [c.hex for c in profile.worst_colors]:
                if item.category in ["tops", "dresses"]:  # Near face
                    color_score -= 2
                    item.color_placement = "avoid_near_face"
                else:
                    item.color_placement = "bottom_only"

            item.color_harmony_score = color_score

        return sorted(items, key=lambda x: x.color_harmony_score, reverse=True)
```

---

### Component 6: Conversational Intelligence

**Natural Styling Dialogue:**

```python
STYLIST_PERSONA = """
You are a warm, knowledgeable personal stylist and best friend. You combine
professional expertise with genuine care for helping people feel confident.

VOICE & TONE:
- Warm but not saccharine
- Confident but not condescending
- Specific but not overwhelming
- Honest but always kind
- Celebratory of their unique beauty

KNOWLEDGE BASE:
- 12-season color analysis (Sci/Art method)
- Body type styling (classic + Kibbe)
- Capsule wardrobe methodology
- Current and timeless fashion
- Occasion-appropriate dressing
- Fashion psychology (enclothed cognition)

COMMUNICATION PRINCIPLES:
1. Always explain WHY something works or doesn't
2. Frame feedback positively (what TO do, not just what not to do)
3. Celebrate their existing good choices
4. Make concrete, actionable suggestions
5. Reference their specific profile data
6. Use their color season terminology naturally

NEVER:
- Use generic advice that could apply to anyone
- Make them feel bad about their body or choices
- Suggest they need to "fix" something about themselves
- Give advice without knowing their profile
- Recommend colors outside their season without noting it
"""

# Response templates that feel human
OUTFIT_FEEDBACK_TEMPLATE = """
**{overall_vibe}** {rating_emoji}

{positive_opener}

**What's Working:**
{working_points}

{color_section}

**To Take It Further:**
{suggestions}

{closing_personalized}
"""

# Example personalized responses
PERSONALIZED_RESPONSES = {
    "color_match": {
        "perfect": "That {color} is *chef's kiss* for your {season} coloring - it makes your {feature} absolutely glow!",
        "good": "The {color} works nicely with your {undertone} undertone.",
        "could_improve": "This {color} is a bit {issue} for your coloring. Try swapping for {suggestion} - it'll be magic!",
    },
    "body_type": {
        "hourglass": "The defined waist here is playing to your strengths beautifully.",
        "pear": "Love how the {top_detail} draws the eye up and balances your silhouette.",
        "rectangle": "The {detail} is creating that waist definition we talked about!",
    }
}
```

---

## User Flows

### Flow 1: New User Onboarding

```
TRIGGER: User sends /start or first message

STEP 1: Welcome + Value Prop
"Hey! I'm your personal AI stylist. I learn your unique coloring, body,
and style to give you advice that's actually personalized to YOU.

Want to discover your colors? Send me a selfie in natural light!"

STEP 2: Color Analysis (Core)
[User sends selfie]
→ Analyze undertone, value, chroma, contrast
→ Determine season with confidence score
→ Present results warmly with "wow" moments
→ Save to profile

STEP 3: Style Preferences (Conversational)
"Now I know your colors! Quick question - which best describes your daily life?
A) Office/corporate most days
B) Casual/creative environment
C) Mix of everything
D) Mostly at home/remote"

[Build lifestyle context through 2-3 natural questions]

STEP 4: First Value
"Based on what I know so far, here are 3 colors you should definitely
have in your closet: [specific to their season]

Send me an outfit photo anytime and I'll tell you how it's working!"
```

### Flow 2: Outfit Analysis

```
TRIGGER: User sends outfit photo (with or without caption)

STEP 1: Analyze Image
→ Detect garments, colors, patterns
→ Assess fit and silhouette
→ Evaluate occasion appropriateness

STEP 2: Cross-Reference Profile
→ Check colors against their season
→ Check silhouette against body type
→ Check formality against lifestyle

STEP 3: Generate Personalized Feedback
"Looking sharp! That navy blazer is perfect for your {season} palette.

✓ The structured shoulders are balancing beautifully
✓ Navy + cream is a classic combo for you
→ Consider swapping the gray pants for chocolate brown -
   it'll be more harmonious with your warm coloring

For your interview: You're projecting confidence and competence.
Add gold earrings (your metal!) and you're set!"

STEP 4: Optional Follow-up
"Want makeup suggestions to complete the look?"
```

### Flow 3: "What Should I Wear?"

```
TRIGGER: User asks what to wear for [occasion]

STEP 1: Gather Context
"A work presentation - exciting! Quick Qs:
- How formal is this? (boardroom vs team meeting)
- Do you want to feel powerful, approachable, or creative?"

STEP 2: Generate Options
[If wardrobe exists, pull from their items]
[If not, suggest based on their palette]

"For a powerful presentation look in YOUR colors:

OPTION 1: The Authority Play
• Your chocolate brown trousers (I know you have these!)
• Cream blouse (soft against your face)
• Teal blazer - this is your POWER color
• Gold jewelry

OPTION 2: The Classic...
[continue]"

STEP 3: Styling Details
"Pro tip: As a {body_type}, the blazer with defined shoulders
will project authority. Tuck the blouse to show your waist."
```

---

## Implementation Phases

### Phase 1: Foundation (Complete Core)
- [x] Basic vision analysis (outfit, colors)
- [x] 12-season color framework
- [x] Style profile data model
- [ ] Expand season palettes (all 12 complete)
- [ ] Improve color analysis prompts
- [ ] Add confidence scoring

### Phase 2: Deep Personalization
- [ ] Body type analysis (classic 5-type)
- [ ] Lifestyle quiz flow
- [ ] Style archetype identification
- [ ] Fit preference learning
- [ ] Feedback loop system

### Phase 3: Wardrobe Intelligence
- [ ] Full wardrobe cataloging
- [ ] Outfit combination engine
- [ ] Gap analysis
- [ ] Capsule wardrobe suggestions
- [ ] "What to wear" from wardrobe

### Phase 4: Makeup & Beauty
- [ ] Full makeup recommendations by season
- [ ] Product suggestions (affiliate potential)
- [ ] Skin concern awareness
- [ ] Event-specific looks

### Phase 5: Advanced Features
- [ ] Kibbe body type (optional advanced)
- [ ] Shopping suggestions with affiliate links
- [ ] Weather integration
- [ ] Calendar/event integration
- [ ] Virtual try-on (future)

---

## Success Metrics

**Engagement:**
- Profile completion rate (target: 60%+ complete color analysis)
- Photos analyzed per user per week
- Return rate (7-day, 30-day)

**Quality:**
- Outfit feedback helpfulness rating
- Color analysis accuracy (user confirmation)
- Recommendation acceptance rate

**Growth:**
- Wardrobe items cataloged per user
- Referral rate (users sharing with friends)

---

## Sources

- [Stitch Fix Algorithms Tour](https://algorithms-tour.stitchfix.com/)
- [Stitch Fix AI Style Assistant](https://newsroom.stitchfix.com/blog/how-were-revolutionizing-personal-styling-with-generative-ai/)
- [Sephora Color IQ](https://digiday.com/marketing/color-iq-sephoras-shade-matching-skin-care-tool-boosts-brand-loyalty/)
- [12 Season Color Analysis](https://radiantlydressed.com/explore-the-12-seasons-of-color/)
- [Professional Color Analyst Standards](https://www.styleacademyintl.com/post/how-to-recognize-a-good-color-analyst-and-why-it-matters)
- [Kibbe Body Types](https://theconceptwardrobe.com/kibbe-body-types/an-introduction-to-the-kibbe-body-types)
- [Capsule Wardrobe Principles](https://capsule.style/en/blogs/capsule-blog-4/capsule-wardrobe-defined)
- [AI Body Shape Analysis](https://blog.looksmaxxreport.com/ai-body-shape-styling-fashion/)
- [Fashion Psychology Research](https://pmc.ncbi.nlm.nih.gov/articles/PMC8455911/)
- [Enclothed Cognition](https://www.foreo.com/mysa/dressed-for-success-how-style-impacts-confidence)
- [Best AI Stylist Apps 2026](https://klodsy.com/blog/best-ai-stylist-apps-2026-comparison/)
