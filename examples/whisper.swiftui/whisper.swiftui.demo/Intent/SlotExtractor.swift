import Foundation
import os.log

struct SlotExtractionResult {
    let slots: [String: Any]
    let confidence: Float
}

class SlotExtractor {
    
    // MARK: - Constants
    
    private static let logger = Logger(subsystem: "com.whispercpp.demo", category: "SlotExtractor")
    
    // MARK: - Slot Templates
    
    private let intentSlotTemplates: [String: [String]] = [
        "QueryPoint": ["metric"],  // time_ref, unit, and identifier can have defaults
        "SetGoal": ["metric", "target"],  // unit can be inferred
        "SetThreshold": ["metric", "threshold"],  // unit can be inferred
        "TimerStopwatch": ["tool", "action"],  // value only required for "set timer X min"
        "ToggleFeature": ["feature", "state"],
        "LogEvent": ["event_type"],  // value, unit, time_ref can have defaults
        "StartActivity": ["activity_type"],
        "StopActivity": ["activity_type"],
        "OpenApp": ["app"],  // action and target can have defaults
        "PhoneAction": ["contact"], // no need of action
        "MediaAction": ["action"],  // target can be optional for some actions
        "WeatherQuery": ["location"],  // attribute can default to general weather
        "QueryTrend": ["metric"],  // period and unit can have defaults
        "IrrelevantInput": []  // No required slots for irrelevant input
    ]
    
    func getRequiredSlots(_ intent: String) -> [String] {
        return intentSlotTemplates[intent] ?? []
    }
    
    // Special method to check if slots are adequate for the specific action within an intent
    func areRequiredSlotsSatisfied(_ intent: String, slots: [String: Any], text: String) -> Bool {
        let baseRequiredSlots = getRequiredSlots(intent)
        
        // Check if all base required slots are present
        if !baseRequiredSlots.allSatisfy({ slots.keys.contains($0) }) {
            return false
        }
        
        // Special case handling for TimerStopwatch - value required only for "set timer"
        if intent == "TimerStopwatch" {
            if let action = slots["action"] as? String,
               let tool = slots["tool"] as? String {
                // If setting a timer, value is required
                if (action == "set" || action == "start") && tool == "timer" {
                    if slots["value"] == nil {
                        return false
                    }
                }
            }
        }
        
        // Special case for SetGoal and SetThreshold - target/threshold must be meaningful
        if intent == "SetGoal" && slots["target"] == nil {
            return false
        }
        
        if intent == "SetThreshold" && slots["threshold"] == nil {
            return false
        }
        
        return true
    }
    
    // MARK: - Synonym Mappings based on Kotlin implementation
    
    private let metricSynonyms: [String: [String]] = [
        "steps": [
            "steps", "step", "walk", "walked", "walking", "footsteps", "pace", "stride",
            "tread", "footfall", "gait", "paces", "stroll", "strolling", "strolled",
            "amble", "ambling", "saunter", "march", "marching", "trudge", "hike",
            "hiking", "trek", "wander", "movement", "activity", "moves"
        ],
        "distance": [
            "distance", "walked", "walk", "miles", "kilometers", "km", "far", "meter",
            "metres", "meters", "travelled", "traveled", "covered", "journey", "range",
            "length", "span", "route", "path", "mileage", "odometer", "how far",
            "feet", "yards"
        ],
        "calories": [
            "calories", "calorie", "kcal", "energy", "burned", "burn", "burning",
            "burnt", "expended", "consumed", "expenditure", "kilojoules", "kj",
            "nutrition", "intake", "food energy", "metabolic", "metabolism",
            "fat burn", "fat burning", "cal"
        ],
        "heart rate": [
            "heart rate", "heartrate", "hr", "pulse", "bpm", "heart beat", "heartbeat",
            "cardiac", "beats per minute", "resting heart rate", "rhr", "heart rhythm",
            "cardiovascular", "cardio", "heart health", "ticker", "heart pulse",
            "pulse rate", "heart monitor", "heart"
        ],
        "sleep": [
            "sleep", "slept", "sleeping", "rest", "rested", "resting", "nap",
            "napping", "napped", "slumber", "snooze", "snoozed", "doze", "dozed",
            "asleep", "bedtime", "night sleep", "shuteye", "zzz", "sleep time",
            "sleep duration", "hours slept", "sleep hours"
        ],
        "sleep score": [
            "sleep score", "sleep quality", "sleep rating", "sleep performance",
            "sleep analysis", "sleep grade", "sleep rank", "sleep level",
            "sleep efficiency", "sleep assessment", "how well slept", "sleep health",
            "sleep metric", "sleep stats", "sleep report", "sleep evaluation"
        ],
        "spo2": [
            "spo2", "oxygen", "blood oxygen", "o2", "saturation", "oxygen saturation",
            "oxygen level", "oxygen levels", "o2 sat", "blood o2", "oxygen sat",
            "pulse ox", "pulse oximetry", "oximeter", "oxygen reading", "o2 level",
            "respiratory", "breathing", "blood oxygen level"
        ],
        "weight": [
            "weight", "weigh", "kg", "pounds", "lbs", "kilogram", "kilograms", "lb",
            "body weight", "mass", "scale", "weighed", "weighing", "bmi",
            "body mass", "how much weigh", "weight reading", "weighted", "grams",
            "stone", "ounces", "oz"
        ],
        "stress": [
            "stress", "stressed", "anxiety", "tension", "anxious", "worried",
            "worry", "pressure", "strain", "overwhelmed", "nervous", "nervousness",
            "stress level", "mental stress", "emotional stress", "burnout",
            "stress score", "relaxation", "calm", "mental health", "wellbeing"
        ]
    ]
    
    private let timeSynonyms: [String: [String]] = [
        "today": [
            "today", "now", "currently", "this day", "present", "right now",
            "at present", "so far today", "today's", "current day", "as of today",
            "till now", "up to now", "at the moment", "presently", "at this time",
            "this very day", "the present day", "nowadays", "in the present",
            "for today", "on this day", "todays", "since midnight"
        ],
        "yesterday": [
            "yesterday", "last day", "previous day", "day before", "1 day ago",
            "one day ago", "a day ago", "the day before", "prior day",
            "the previous day", "24 hours ago", "yesterdays", "yesterday's",
            "the other day", "day prior", "the last day", "past day",
            "the day that was", "the preceding day", "most recent day",
            "the latest day", "just yesterday", "only yesterday", "back yesterday"
        ],
        "last night": [
            "last night", "night", "overnight", "during sleep", "while sleeping",
            "nighttime", "night time", "at night", "during the night",
            "throughout the night", "all night", "past night", "previous night",
            "the other night", "last evening", "yesterday night", "yesterday evening",
            "late yesterday", "after dark", "hours of sleep", "sleeping hours",
            "bedtime", "sleep time", "in bed", "whilst asleep", "sleep period"
        ],
        "this morning": [
            "this morning", "morning", "am", "early today", "earlier today",
            "this am", "today morning", "in the morning", "morning time",
            "mornings", "early hours", "before noon", "dawn", "daybreak",
            "sunrise", "first thing", "early on", "at dawn", "morning hours",
            "start of day", "beginning of day", "waking hours", "after waking",
            "upon waking", "since waking"
        ],
        "this week": [
            "this week", "current week", "weekly", "so far this week",
            "week to date", "wtd", "the week", "present week", "the current week",
            "in this week", "for the week", "throughout the week", "during the week",
            "over the week", "7 days", "past 7 days", "last 7 days",
            "these 7 days", "this weeks", "this week's", "since monday",
            "week so far", "till now this week", "up to now this week", "weekly total"
        ],
        "last week": [
            "last week", "past week", "previous week", "the week before",
            "prior week", "1 week ago", "one week ago", "a week ago",
            "week prior", "the last week", "the past week", "the previous week",
            "7 days ago", "last weeks", "last week's", "the preceding week",
            "most recent week", "latest week", "former week", "earlier week",
            "the other week", "back last week", "during last week", "throughout last week",
            "over last week"
        ],
        "this month": [
            "this month", "current month", "monthly", "so far this month",
            "month to date", "mtd", "the month", "present month",
            "the current month", "in this month", "for the month",
            "throughout the month", "during the month", "over the month",
            "30 days", "past 30 days", "last 30 days", "these 30 days",
            "this months", "this month's", "since the 1st", "month so far",
            "till now this month", "up to now this month", "monthly total"
        ]
    ]
    
    private let qualifierSynonyms: [String: [String]] = [
        "minimum": [
            "minimum", "min", "lowest", "least", "bottom", "bare minimum",
            "minimal", "minimally", "rock bottom", "floor", "base", "baseline",
            "low point", "lower", "smallest", "tiniest", "fewest", "less",
            "lesser", "reduced", "at least", "no less than", "starting from",
            "beginning at", "from", "low", "lows", "worst", "slowest"
        ],
        "maximum": [
            "maximum", "max", "highest", "most", "peak", "top", "maximal",
            "maximally", "ceiling", "upper limit", "high point", "higher",
            "greatest", "largest", "biggest", "best", "record", "all time high",
            "at most", "no more than", "up to", "limit", "cap", "high", "highs",
            "fastest", "extreme", "topmost", "ultimate"
        ],
        "average": [
            "average", "avg", "mean", "typical", "normal", "averaged", "averaging",
            "median", "mid", "middle", "midpoint", "central", "moderate",
            "standard", "regular", "usual", "common", "ordinary", "per day",
            "daily average", "on average", "typically", "normally", "generally",
            "approximately", "around", "about", "roughly"
        ],
        "total": [
            "total", "sum", "overall", "complete", "entire", "totaled", "totaling",
            "all", "all time", "full", "whole", "combined", "cumulative",
            "aggregate", "collectively", "together", "in total", "altogether",
            "grand total", "summation", "net", "gross", "comprehensive",
            "accumulated", "compilation", "tally", "count", "running total"
        ]
    ]
    
    // Pre-compiled regex patterns for better performance
    private let walkingMovementRegex = try! NSRegularExpression(pattern: "\\b(?:walk|walked|walking|stroll|strolled|strolling|hike|hiked|hiking|trek|trekked|trekking|march|marched|marching|wander|wandered|wandering|roam|roamed|roaming|amble|ambled|ambling|saunter|sauntered|sauntering|trudge|trudged|trudging|move|moved|moving|movement)\\b", options: [.caseInsensitive])
    private let distanceRegex = try! NSRegularExpression(pattern: "\\b(?:far|distance|km|kilometers|kilometre|kilometres|mile|miles|meter|meters|metre|metres|feet|ft|yard|yards|yd|long|length|covered|travelled|traveled|route|path|journey|span|range|how far|mileage)\\b", options: [.caseInsensitive])
    private let sleepRegex = try! NSRegularExpression(pattern: "\\b(?:sleep|slept|sleeping|asleep|nap|napped|napping|rest|rested|resting|snooze|snoozed|snoozing|doze|dozed|dozing|slumber|bedtime|night|overnight|bed|zzz)\\b", options: [.caseInsensitive])
    private let sleepQualityRegex = try! NSRegularExpression(pattern: "\\b(?:quality|score|rating|rate|well|badly|good|bad|poor|deep|light|efficiency|grade|rank|analysis|performance|how well)\\b", options: [.caseInsensitive])
    private let heartRegex = try! NSRegularExpression(pattern: "\\b(?:heart|cardiac|cardio|cardiovascular|pulse|beat|beats|beating|bpm|rhythm|ticker)\\b", options: [.caseInsensitive])
    private let caloriesRegex = try! NSRegularExpression(pattern: "\\b(?:calorie|calories|kcal|energy|burn|burned|burnt|burning|expend|expended|consume|consumed|intake|kilojoule|kilojoules|kj|food energy|metabolic|metabolism|fat)\\b", options: [.caseInsensitive])
    private let oxygenRegex = try! NSRegularExpression(pattern: "\\b(?:oxygen|o2|spo2|saturation|sat|blood oxygen|pulse ox|oximeter|oximetry|breathing|respiratory|respiration|air|breathe)\\b", options: [.caseInsensitive])
    private let weightRegex = try! NSRegularExpression(pattern: "\\b(?:weight|weigh|weighing|weighed|kg|kilogram|kilograms|pound|pounds|lbs|lb|body mass|bmi|body weight|mass|scale|heavy|light|stone|gram|grams|ounce|ounces|oz)\\b", options: [.caseInsensitive])
    private let stressRegex = try! NSRegularExpression(pattern: "\\b(?:stress|stressed|stressful|anxiety|anxious|tension|tense|worried|worry|worrying|pressure|pressured|strain|strained|overwhelm|overwhelmed|nervous|nervousness|burnout|mental health|relaxation|relax|calm|peace|peaceful)\\b", options: [.caseInsensitive])
    private let heartRateUnitRegex = try! NSRegularExpression(pattern: "\\b(?:heart\\s+rate|pulse|hr)\\b", options: [.caseInsensitive])
    private let weightUnitRegex = try! NSRegularExpression(pattern: "\\b(?:weight|weigh)\\b", options: [.caseInsensitive])
    private let stepsUnitRegex = try! NSRegularExpression(pattern: "\\bsteps?\\b", options: [.caseInsensitive])
    private let numberRegex = try! NSRegularExpression(pattern: "\\b(\\d+(?:\\.\\d+)?)\\b", options: [])
    private let numberSequenceRegex = try! NSRegularExpression(pattern: "\\b(\\d+(?:\\.\\d+)?)\\b", options: [])
    
    // Pre-compiled synonym patterns for better performance
    private var synonymPatterns: [String: NSRegularExpression] = [:]
    
    override init() {
        super.init()
        // Pre-compile synonym patterns
        for (metric, synonyms) in metricSynonyms {
            let escapedSynonyms = synonyms.map { NSRegularExpression.escapedPattern(for: $0) }
            let pattern = "\\b(?:\(escapedSynonyms.joined(separator: "|")))\\b"
            synonymPatterns[metric] = try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        }
    }
    
    // Pre-compiled unit regex patterns
    private let unitPatterns: [String: NSRegularExpression] = [
        "bpm": try! NSRegularExpression(pattern: "\\b(?:bpm|beats?\\s+per\\s+minute|heart\\s+rate|pulse\\s+rate|hr|heartbeat|heart\\s+beat|pulse|cardiac\\s+rate|beat\\s+rate|rhythm|heart\\s+rhythm|cardiac\\s+rhythm)\\b", options: [.caseInsensitive]),
        "kg": try! NSRegularExpression(pattern: "\\b(?:kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|k\\.?g\\.?)\\b", options: [.caseInsensitive]),
        "pounds": try! NSRegularExpression(pattern: "\\b(?:pounds?|lbs?|lb|pound\\s+weight|#|lbs\\s+weight|lb\\s+weight)\\b", options: [.caseInsensitive]),
        "km": try! NSRegularExpression(pattern: "\\b(?:km|kms|kilometer|kilometers|kilometre|kilometres|k\\.?m\\.?|klick|klicks)\\b", options: [.caseInsensitive]),
        "miles": try! NSRegularExpression(pattern: "\\b(?:miles?|mi|mile\\s+distance|mi\\.?|statute\\s+miles?)\\b", options: [.caseInsensitive]),
        "kcal": try! NSRegularExpression(pattern: "\\b(?:kcal|calories?|calorie|cal|cals|kilocalories?|kilocalorie|food\\s+calories?|dietary\\s+calories?|energy|k\\.?cal)\\b", options: [.caseInsensitive]),
        "hours": try! NSRegularExpression(pattern: "\\b(?:hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?)\\b", options: [.caseInsensitive]),
        "minutes": try! NSRegularExpression(pattern: "\\b(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?)\\b", options: [.caseInsensitive]),
        "percent": try! NSRegularExpression(pattern: "\\b(?:percent|%|percentage|pct|pc|per\\s+cent|percentile|blood oxygen|spo2|sp2|spO2)\\b", options: [.caseInsensitive]),
        "count": try! NSRegularExpression(pattern: "\\b(?:steps?|step\\s+count|footsteps?|foot\\s+steps?|paces?|strides?|walk\\s+count|walking\\s+count|number\\s+of\\s+steps?|total\\s+steps?|step\\s+total)\\b", options: [.caseInsensitive]),
        "meters": try! NSRegularExpression(pattern: "\\b(?:meters?|metres?|m|meter\\s+distance|metre\\s+distance|m\\.?)\\b", options: [.caseInsensitive]),
        "feet": try! NSRegularExpression(pattern: "\\b(?:feet|foot|ft|f\\.?t\\.?|')\\b", options: [.caseInsensitive]),
        "seconds": try! NSRegularExpression(pattern: "\\b(?:seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?)\\b", options: [.caseInsensitive]),
        "grams": try! NSRegularExpression(pattern: "\\b(?:grams?|grammes?|g|gm|gms|g\\.?m?\\.?s?\\.?)\\b", options: [.caseInsensitive]),
        "liters": try! NSRegularExpression(pattern: "\\b(?:liters?|litres?|l|ltr|ltrs|l\\.?t?r?\\.?s?\\.?)\\b", options: [.caseInsensitive]),
        "degrees": try! NSRegularExpression(pattern: "\\b(?:degrees?|deg|°|degree\\s+celsius|degree\\s+fahrenheit|celsius|fahrenheit)\\b", options: [.caseInsensitive]),
        "score": try! NSRegularExpression(pattern: "\\b(?:score|rating|grade|rank|level|point|points|pts?|value|number)\\b", options: [.caseInsensitive]),
        "distance": try! NSRegularExpression(pattern: "\\b(?:distance|length|span|range|mileage|how\\s+far|travelled|traveled|covered)\\b", options: [.caseInsensitive])
    ]
    
    
    func extractSlots(text: String, intent: String) -> SlotExtractionResult {
        Self.logger.info("🏷️ Extracting slots for intent: \(intent), text: '\(text)'")
        
        var slots: [String: Any] = [:]
        let textLower = text.lowercased()
        
        // Get required slots for this intent
        let requiredSlots = intentSlotTemplates[intent] ?? []
        
        // Pre-process text
        let processedText = preprocessText(textLower)
        
        // Extract each required slot
        for slotName in requiredSlots {
            if let value = extractSingleSlot(processedText: processedText, originalText: textLower, slotName: slotName, intent: intent) {
                slots[slotName] = value
                Self.logger.info("  ✓ Extracted \(slotName): \(value)")
            }
        }
        
        // Special handling for SetThreshold: combine type into metric
        if intent == "SetThreshold" {
            if let type = extractSingleSlot(processedText: processedText, originalText: textLower, slotName: "type", intent: intent) as? String,
               let metric = slots["metric"] as? String {
                // Set unit based on base metric before combining
                if slots["unit"] == nil {
                    switch metric {
                    case "steps":
                        slots["unit"] = "count"
                    case "distance":
                        slots["unit"] = "km"
                    case "calories":
                        slots["unit"] = "kcal"
                    case "heart rate":
                        slots["unit"] = "bpm"
                    case "sleep":
                        slots["unit"] = "hours"
                    case "weight":
                        slots["unit"] = "kg"
                    case "spo2":
                        slots["unit"] = "percent"
                    case "stress":
                        slots["unit"] = "score"
                    default:
                        break
                    }
                }
                slots["metric"] = "\(type) \(metric)"
                // Self.logger.info("  ✓ Combined metric: \(slots["metric"]!)")
            }
        }
        
        // Add contextual slots
        addContextualSlots(text: textLower, intent: intent, slots: &slots)
        
        // Calculate confidence based on how many slots were extracted
        let confidence: Float = requiredSlots.isEmpty ? 1.0 : Float(slots.count) / Float(requiredSlots.count)
        
        Self.logger.info("🎯 Final slots: \(slots) (confidence: \(String(format: "%.2f", confidence)))")
        
        return SlotExtractionResult(slots: slots, confidence: confidence)
    }    private func preprocessText(_ text: String) -> String {
        var processed = text
        
        // Normalize common variations (from Python implementation)
        processed = processed.replacingOccurrences(of: "\\bhow\\s+much\\s+did\\s+i\\s+walk", with: "walking distance", options: .regularExpression)
        processed = processed.replacingOccurrences(of: "\\bhow\\s+many\\s+steps", with: "steps", options: .regularExpression)
        processed = processed.replacingOccurrences(of: "\\bhow\\s+far\\s+did\\s+i\\s+walk", with: "walking distance", options: .regularExpression)
        processed = processed.replacingOccurrences(of: "\\bwhat\\s+is\\s+my", with: "my", options: .regularExpression)
        processed = processed.replacingOccurrences(of: "\\bshow\\s+me\\s+my", with: "my", options: .regularExpression)
        
        return processed
    }
    
    private func extractSingleSlot(processedText: String, originalText: String, slotName: String, intent: String) -> Any? {
        switch slotName {
        case "metric":
            return extractMetric(processedText: processedText, originalText: originalText)
        case "time_ref":
            return extractTimeRef(text: originalText)
        case "unit":
            return extractUnit(text: originalText)
        case "qualifier":
            return extractQualifier(text: originalText)
        case "identifier":
            return extractIdentifier(text: originalText)
        case "threshold":
            return extractThreshold(text: originalText)
        case "target":
            return extractTarget(text: originalText)
        case "value":
            return extractValue(text: originalText, intent: intent)
        case "feature":
            return extractFeature(text: originalText)
        case "state":
            return extractState(text: originalText)
        case "action":
            return extractAction(text: originalText)
        case "tool":
            return extractTool(text: originalText)
        case "activity_type":
            return extractActivityType(text: originalText)
        case "app":
            return extractApp(text: originalText)
        case "contact":
            return extractContact(text: originalText)
        case "location":
            return extractLocation(text: originalText)
        case "attribute":
            return extractAttribute(text: originalText)
        case "type":
            return extractType(text: originalText)
        case "period":
            return extractPeriod(text: originalText)
        case "event_type":
            return extractEventType(text: originalText)
        case "message":
            return "Sorry, please say again"
        default:
            return nil
        }
    }
    
    
    private func extractMetric(processedText: String, originalText: String) -> String? {
        // Direct synonym matching on processed text first
        for (metric, pattern) in synonymPatterns {
            if pattern.numberOfMatches(in: processedText, options: [], range: NSRange(location: 0, length: processedText.count)) > 0 {
                return metric
            }
        }
        
        // Fallback to original text
        for (metric, pattern) in synonymPatterns {
            if pattern.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
                return metric
            }
        }
        
        // Context-based inference with expanded patterns
        
        // Walking/Movement context
        if walkingMovementRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return distanceRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 ? "distance" : "steps"
        }
        
        // Sleep context
        if sleepRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return sleepQualityRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 ? "sleep score" : "sleep"
        }
        
        // Heart/Cardio context
        if heartRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return "heart rate"
        }
        
        // Calories/Energy context
        if caloriesRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return "calories"
        }
        
        // Oxygen/Breathing context
        if oxygenRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return "spo2"
        }
        
        // Weight/Body Mass context
        if weightRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return "weight"
        }
        
        // Stress/Mental Health context
        if stressRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return "stress"
        }
        
        return nil
    }
    
    private func extractTimeRef(text: String) -> String? {
        let timePatterns: [String: String] = [
            "last night": "\\blast\\s+night\\b|\\bduring\\s+(?:the\\s+)?night\\b|\\bovernight\\b|\\bnight\\s+time\\b|\\bnighttime\\b|\\bat\\s+night\\b|\\bthrough(?:out)?\\s+(?:the\\s+)?night\\b|\\ball\\s+night\\b|\\bpast\\s+night\\b|\\bprevious\\s+night\\b|\\bthe\\s+other\\s+night\\b|\\blast\\s+evening\\b|\\byesterday\\s+night\\b|\\byesterday\\s+evening\\b|\\blate\\s+yesterday\\b|\\bafter\\s+dark\\b|\\bwhile\\s+(?:I\\s+)?sleep(?:ing)?\\b|\\bduring\\s+(?:my\\s+)?sleep\\b|\\bsleep(?:ing)?\\s+(?:hours|time|period)\\b|\\bin\\s+bed\\b|\\bwhilst\\s+asleep\\b|\\bwhen\\s+(?:I\\s+)?(?:was\\s+)?asleep\\b|\\bbedtime\\b",
            
            "yesterday": "\\byesterday\\b(?!\\s+(?:night|evening|morning|afternoon))|\\blast\\s+day\\b|\\bprevious\\s+day\\b|\\bday\\s+before\\b|\\b1\\s+day\\s+ago\\b|\\bone\\s+day\\s+ago\\b|\\ba\\s+day\\s+ago\\b|\\bthe\\s+day\\s+before\\b|\\bprior\\s+day\\b|\\bthe\\s+previous\\s+day\\b|\\b24\\s+hours\\s+ago\\b|\\byesterdays?\\b|\\bthe\\s+other\\s+day\\b|\\bday\\s+prior\\b|\\bthe\\s+last\\s+day\\b|\\bpast\\s+day\\b|\\bthe\\s+preceding\\s+day\\b|\\bmost\\s+recent\\s+day\\b|\\bjust\\s+yesterday\\b|\\bonly\\s+yesterday\\b|\\bback\\s+yesterday\\b|\\byesterday\\s+morning\\b|\\byesterday\\s+am\\b|\\bmorning\\s+yesterday\\b|\\byesterday\\s+in\\s+the\\s+morning\\b|\\byesterday\\s+early\\b|\\byesterday\\s+at\\s+dawn\\b|\\bearly\\s+yesterday\\b|\\byesterday\\s+daybreak\\b|\\byesterday\\s+sunrise\\b|\\byesterday\\s+first\\s+thing\\b|\\byesterday\\s+afternoon\\b|\\byesterday\\s+pm\\b|\\bafternoon\\s+yesterday\\b|\\byesterday\\s+in\\s+the\\s+afternoon\\b|\\byesterday\\s+midday\\b|\\byesterday\\s+noon\\b|\\byesterday\\s+lunchtime\\b|\\byesterday\\s+mid[\\s-]?day\\b|\\byesterday\\s+evening\\b|\\bevening\\s+yesterday\\b|\\byesterday\\s+in\\s+the\\s+evening\\b|\\byesterday\\s+at\\s+night\\b|\\blate\\s+yesterday\\b|\\byesterday\\s+dusk\\b|\\byesterday\\s+twilight\\b|\\byesterday\\s+sundown\\b|\\byesterday\\s+nightfall\\b",
            
            "now": "\\bnow\\b|\\bright\\s+now\\b|\\bat\\s+(?:this\\s+)?moment\\b|\\bthis\\s+instant\\b|\\bimmediately\\b|\\binstantly\\b|\\bright\\s+away\\b|\\bright\\s+at\\s+this\\s+moment\\b|\\bat\\s+the\\s+current\\s+time\\b|\\bthis\\s+very\\s+moment\\b|\\bcurrent\\s+time\\b|\\bthe\\s+present\\s+moment\\b|\\bright\\s+here\\b|\\bright\\s+at\\s+this\\s+instant\\b",
            
            "today": "\\btoday\\b(?!\\s+(?:morning|afternoon|evening|night))|\\bcurrently\\b|\\bthis\\s+day\\b|\\bat\\s+present\\b|\\bso\\s+far\\s+today\\b|\\btodays?\\b|\\bcurrent\\s+day\\b|\\bas\\s+of\\s+today\\b|\\btill\\s+now\\b|\\bup\\s+to\\s+now\\b|\\bpresently\\b|\\bat\\s+this\\s+time\\b|\\bthis\\s+very\\s+day\\b|\\bthe\\s+present\\s+day\\b|\\bfor\\s+today\\b|\\bon\\s+this\\s+day\\b|\\bsince\\s+midnight\\b|\\bso\\s+far\\b|\\buntil\\s+now\\b|\\bas\\s+of\\s+now\\b|\\blater\\s+today\\b|\\bend\\s+of\\s+(?:the\\s+)?day\\b",
            
            "last week": "\\blast\\s+week\\b|\\bpast\\s+week\\b|\\bprevious\\s+week\\b|\\bthe\\s+week\\s+before\\b|\\bprior\\s+week\\b|\\b1\\s+week\\s+ago\\b|\\bone\\s+week\\s+ago\\b|\\ba\\s+week\\s+ago\\b|\\bweek\\s+prior\\b|\\bthe\\s+last\\s+week\\b|\\bthe\\s+past\\s+week\\b|\\bthe\\s+previous\\s+week\\b|\\b7\\s+days\\s+ago\\b|\\blast\\s+weeks?\\b|\\bthe\\s+preceding\\s+week\\b|\\bmost\\s+recent\\s+week\\b|\\blatest\\s+week\\b|\\bformer\\s+week\\b|\\bearlier\\s+week\\b|\\bthe\\s+other\\s+week\\b|\\bduring\\s+last\\s+week\\b|\\bthroughout\\s+last\\s+week\\b|\\bover\\s+last\\s+week\\b|\\bback\\s+last\\s+week\\b",
            
            "tomorrow": "\\btomorrow\\b|\\bnext\\s+day\\b|\\bthe\\s+day\\s+after\\b|\\bday\\s+after\\b|\\btomorrow\\s+morning\\b|\\btomorrow\\s+afternoon\\b|\\btomorrow\\s+evening\\b|\\btomorrow\\s+night\\b|\\bcoming\\s+day\\b|\\bupcoming\\s+day\\b|\\bfuture\\s+day\\b|\\bthe\\s+following\\s+day\\b|\\b24\\s+hours\\s+from\\s+now\\b|\\bin\\s+24\\s+hours\\b|\\bby\\s+tomorrow\\b|\\btill\\s+tomorrow\\b|\\buntil\\s+tomorrow\\b",
            
            "this morning": "\\bthis\\s+morning\\b|\\bmorning\\b(?!\\s+(?:yesterday|tomorrow|next|last|this\\s+(?:afternoon|evening|week|month|year)))|\\bearly\\s+today\\b|\\btoday\\s+morning\\b|\\bin\\s+the\\s+morning\\b|\\bthis\\s+am\\b|\\bearly\\s+hours\\b|\\bdawn\\b|\\bsunrise\\b|\\bdaybreak\\b|\\bfirst\\s+thing\\b|\\bright\\s+after\\s+waking\\b|\\bupon\\s+waking\\b|\\bsince\\s+waking\\b",
            
            "this afternoon": "\\bthis\\s+afternoon\\b|\\bafternoon\\b(?!\\s+(?:yesterday|tomorrow|next|last|this\\s+(?:morning|evening|week|month|year)))|\\btoday\\s+afternoon\\b|\\bin\\s+the\\s+afternoon\\b|\\bthis\\s+pm\\b|\\bafter\\s+noon\\b|\\bmidday\\b|\\bmid[\\s-]?day\\b|\\bnoon\\b|\\blunchtime\\b|\\blate\\s+morning\\b|\\bearly\\s+afternoon\\b",
            
            "this evening": "\\bthis\\s+evening\\b|\\bevening\\b(?!\\s+(?:yesterday|tomorrow|next|last|this\\s+(?:morning|afternoon|week|month|year)))|\\btonight\\b|\\btoday\\s+evening\\b|\\bin\\s+the\\s+evening\\b|\\blater\\s+today\\b|\\bend\\s+of\\s+day\\b|\\bafter\\s+work\\b|\\bdusk\\b|\\btwilight\\b|\\bsundown\\b|\\bsunset\\b|\\bnightfall\\b|\\bafter\\s+dark\\b",
            
            "this week": "\\bthis\\s+week\\b|\\bcurrent\\s+week\\b|\\bweek\\s+so\\s+far\\b|\\btill\\s+now\\s+this\\s+week\\b|\\bup\\s+to\\s+now\\s+this\\s+week\\b|\\bweekly\\s+total\\b|\\bweek\\s+to\\s+date\\b|\\bthis\\s+weeks?\\b|\\bongoing\\s+week\\b|\\bpresent\\s+week\\b|\\bwithin\\s+this\\s+week\\b|\\bduring\\s+this\\s+week\\b|\\bthroughout\\s+this\\s+week\\b|\\bover\\s+this\\s+week\\b",
            
            "this month": "\\bthis\\s+month\\b|\\bcurrent\\s+month\\b|\\bmonth\\s+so\\s+far\\b|\\btill\\s+now\\s+this\\s+month\\b|\\bup\\s+to\\s+now\\s+this\\s+month\\b|\\bmonthly\\s+total\\b|\\bmonth\\s+to\\s+date\\b|\\bthis\\s+months?\\b|\\bongoing\\s+month\\b|\\bpresent\\s+month\\b|\\bwithin\\s+this\\s+month\\b|\\bduring\\s+this\\s+month\\b|\\bthroughout\\s+this\\s+month\\b|\\bover\\s+this\\s+month\\b",
            
            "last month": "\\blast\\s+month\\b|\\bpast\\s+month\\b|\\bprevious\\s+month\\b|\\bthe\\s+month\\s+before\\b|\\bprior\\s+month\\b|\\b1\\s+month\\s+ago\\b|\\bone\\s+month\\s+ago\\b|\\ba\\s+month\\s+ago\\b|\\bmonth\\s+prior\\b|\\bthe\\s+last\\s+month\\b|\\bthe\\s+past\\s+month\\b|\\bthe\\s+previous\\s+month\\b|\\blast\\s+months?\\b|\\bthe\\s+preceding\\s+month\\b|\\bmost\\s+recent\\s+month\\b|\\blatest\\s+month\\b|\\bformer\\s+month\\b|\\bearlier\\s+month\\b|\\bthe\\s+other\\s+month\\b|\\bduring\\s+last\\s+month\\b|\\bthroughout\\s+last\\s+month\\b|\\bover\\s+last\\s+month\\b|\\bback\\s+last\\s+month\\b",
            
            "next week": "\\bnext\\s+week\\b|\\bcoming\\s+week\\b|\\bupcoming\\s+week\\b|\\bfollowing\\s+week\\b|\\bweek\\s+ahead\\b|\\bthe\\s+next\\s+7\\s+days\\b|\\bin\\s+a\\s+week\\b|\\ba\\s+week\\s+from\\s+now\\b|\\b7\\s+days\\s+from\\s+now\\b|\\bnext\\s+weeks?\\b|\\bfuture\\s+week\\b|\\bforthcoming\\s+week\\b",
            
            "next month": "\\bnext\\s+month\\b|\\bcoming\\s+month\\b|\\bupcoming\\s+month\\b|\\bfollowing\\s+month\\b|\\bmonth\\s+ahead\\b|\\bthe\\s+next\\s+30\\s+days\\b|\\bin\\s+a\\s+month\\b|\\ba\\s+month\\s+from\\s+now\\b|\\b30\\s+days\\s+from\\s+now\\b|\\bnext\\s+months?\\b|\\bfuture\\s+month\\b|\\bforthcoming\\s+month\\b",
            
            "recently": "\\brecently\\b|\\blately\\b|\\bof\\s+late\\b|\\bin\\s+recent\\s+times\\b|\\bthese\\s+days\\b|\\bthe\\s+past\\s+few\\s+days\\b|\\bthe\\s+last\\s+few\\s+days\\b|\\brecent\\s+days\\b|\\bjust\\s+recently\\b|\\bnot\\s+long\\s+ago\\b|\\ba\\s+short\\s+while\\s+ago\\b|\\bwithin\\s+the\\s+past\\s+few\\s+days\\b|\\bover\\s+the\\s+past\\s+few\\s+days\\b",
            
            "all time": "\\ball\\s+time\\b|\\ball-time\\b|\\bever\\b|\\ball\\s+history\\b|\\bsince\\s+(?:the\\s+)?beginning\\b|\\bfrom\\s+(?:the\\s+)?start\\b|\\bthroughout\\s+history\\b|\\blifetime\\b|\\ball\\s+my\\s+life\\b|\\bsince\\s+I\\s+(?:started|begun|began)\\b|\\bfrom\\s+day\\s+one\\b|\\bsince\\s+inception\\b|\\ball\\s+records\\b|\\ball\\s+data\\b|\\bcomplete\\s+history\\b|\\bfull\\s+history\\b",
            
            "this year": "\\bthis\\s+year\\b|\\bcurrent\\s+year\\b|\\byear\\s+so\\s+far\\b|\\btill\\s+now\\s+this\\s+year\\b|\\bup\\s+to\\s+now\\s+this\\s+year\\b|\\byearly\\s+total\\b|\\byear\\s+to\\s+date\\b|\\bthis\\s+years?\\b|\\bongoing\\s+year\\b|\\bpresent\\s+year\\b|\\bwithin\\s+this\\s+year\\b|\\bduring\\s+this\\s+year\\b|\\bthroughout\\s+this\\s+year\\b|\\bover\\s+this\\s+year\\b",
            
            "last year": "\\blast\\s+year\\b|\\bpast\\s+year\\b|\\bprevious\\s+year\\b|\\bthe\\s+year\\s+before\\b|\\bprior\\s+year\\b|\\b1\\s+year\\s+ago\\b|\\bone\\s+year\\s+ago\\b|\\ba\\s+year\\s+ago\\b|\\byear\\s+prior\\b|\\bthe\\s+last\\s+year\\b|\\bthe\\s+past\\s+year\\b|\\bthe\\s+previous\\s+year\\b|\\blast\\s+years?\\b|\\bthe\\s+preceding\\s+year\\b|\\bmost\\s+recent\\s+year\\b|\\blatest\\s+year\\b|\\bformer\\s+year\\b|\\bearlier\\s+year\\b|\\bthe\\s+other\\s+year\\b|\\bduring\\s+last\\s+year\\b|\\bthroughout\\s+last\\s+year\\b|\\bover\\s+last\\s+year\\b|\\bback\\s+last\\s+year\\b"
        ]
        
        for (timeRef, pattern) in timePatterns {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return timeRef
            }
        }
        
        return nil
    }
    
    private func extractQualifier(text: String) -> String? {
        let qualifierPatterns: [String: String] = [
            "minimum": "\\b(?:minimum|min|lowest|least|bottom|smallest|bare minimum|minimal|minimally|rock bottom|floor|base|baseline|low point|lower|tiniest|fewest|less|lesser|reduced|at least|no less than|starting from|beginning at|from|low|lows|worst|slowest|minimum value|min value|floor value|bottom line|rock-bottom|absolute minimum|very least|bare min)\\b",
            "maximum": "\\b(?:maximum|max|highest|most|peak|top|largest|biggest|maximal|maximally|ceiling|upper limit|high point|higher|greatest|best|record|all time high|at most|no more than|up to|limit|cap|high|highs|fastest|extreme|topmost|ultimate|max value|maximum value|ceiling value|top line|all-time high|absolute maximum|very most|max out)\\b",
            "average": "\\b(?:average|avg|mean|typical|normal|averaged|averaging|median|mid|middle|midpoint|central|moderate|standard|regular|usual|common|ordinary|per day|daily average|on average|typically|normally|generally|approximately|around|about|roughly|average value|mean value|avg value|in average|on avg|medium|middling|fair)\\b",
            "total": "\\b(?:total|sum|overall|complete|entire|all|totaled|totaling|all time|full|whole|combined|cumulative|aggregate|collectively|together|in total|altogether|grand total|summation|net|gross|comprehensive|accumulated|compilation|tally|count|running total|total value|sum total|total amount|cumulative total|overall total|combined total|full total|net total)\\b"
        ]
        
        for (qualifier, pattern) in qualifierPatterns {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return qualifier
            }
        }
        
        return "today"
    }
    
    private func extractIdentifier(text: String) -> String? {
        let identifierPatterns: [String: String] = [
            "minimum": "\\b(?:minimum|min|lowest|least|bottom|bare minimum|minimal|minimally|rock bottom|floor|base|baseline|low point|lower|smallest|tiniest|fewest|less|lesser|reduced|at least|no less than|starting from|beginning at|from|low|lows|worst|slowest)\\b",
            "maximum": "\\b(?:maximum|max|highest|most|peak|top|maximal|maximally|ceiling|upper limit|high point|higher|greatest|largest|biggest|best|record|all time high|at most|no more than|up to|limit|cap|high|highs|fastest|extreme|topmost|ultimate)\\b",
            "average": "\\b(?:average|avg|mean|typical|normal|averaged|averaging|median|mid|middle|midpoint|central|moderate|standard|regular|usual|common|ordinary|per day|daily average|on average|typically|normally|generally|approximately|around|about|roughly)\\b",
            "total": "\\b(?:total|sum|overall|complete|entire|totaled|totaling|all|all time|full|whole|combined|cumulative|aggregate|collectively|together|in total|altogether|grand total|summation|net|gross|comprehensive|accumulated|compilation|tally|count|running total)\\b"
        ]
        
        for (identifier, pattern) in identifierPatterns {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return identifier
            }
        }
        
        // Default to average if no specific identifier is found
        return "average"
    }
    
    private func extractUnit(text: String) -> String? {
        // Try context-based unit inference first
        if heartRateUnitRegex.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
            return "bpm"
        }
        
        if stressRegex.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
            return "score"
        }
        
        if oxygenRegex.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
            return "percent"
        }
        
        if sleepRegex.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
            return "hours"
        }
        
        if sleepQualityRegex.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
            return "score"
        }
        
        if distanceRegex.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
            return "distance"
        }
        
        if caloriesRegex.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
            return "kcal"
        }
        
        if walkingMovementRegex.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
            return "distance"
        }
        
        if weightUnitRegex.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
            return "kg"
        }
        
        if stepsUnitRegex.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
            return "count"
        }
        
        // Check pre-compiled unit patterns
        for (unit, pattern) in unitPatterns {
            if pattern.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
                return unit
            }
        }
        
        return nil
    }
    
    
    private func extractThreshold(text: String) -> Int? {
        // Look for numbers in context
        let numberPatterns = [
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:bpm|beats?\\s+per\\s+minute|kg|kgs|kilogram|kilograms|pounds?|lbs?|lb|km|kms|kilometer|kilometers|kilometre|kilometres|miles?|mi|meter|meters|metre|metres|m|feet|foot|ft|percent|%|percentage|hours?|hrs?|hr|h|minutes?|mins?|min|seconds?|secs?|sec|s|kcal|calories?|cal|cals|steps?|grams?|g|liters?|litres?|l|ltr)\\b",
            "\\b(?:above|over|exceeds?|exceeded|exceeding|higher\\s+than|more\\s+than|greater\\s+than|beyond|past|upwards?\\s+of|in\\s+excess\\s+of|surpass(?:es|ed|ing)?|top(?:s|ped|ping)?|beat(?:s|ing)?|outperform(?:s|ed|ing)?|at\\s+least|minimum\\s+of|no\\s+less\\s+than|starting\\s+from|from|upward\\s+of)\\s+(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:below|under|less\\s+than|lower\\s+than|fewer\\s+than|beneath|short\\s+of|shy\\s+of|down\\s+to|up\\s+to|no\\s+more\\s+than|at\\s+most|maximum\\s+of|capped\\s+at|limited\\s+to|within|not\\s+exceeding|doesn'?t\\s+exceed|under\\s+the|below\\s+the|inferior\\s+to)\\s+(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:around|about|approximately|roughly|nearly|close\\s+to|near|almost|circa|approx\\.?|~|somewhere\\s+around|in\\s+the\\s+region\\s+of|in\\s+the\\s+ballpark\\s+of|give\\s+or\\s+take|or\\s+so|ish)\\s+(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:between|from)\\s+(\\d+(?:\\.\\d+)?)\\s+(?:to|and|through|-|–)\\s+(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:exactly|precisely|just|only|specifically|right\\s+at|dead\\s+on|on\\s+the\\s+dot|bang\\s+on|spot\\s+on)\\s+(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?)\\s+(?:or\\s+(?:more|less|above|below|over|under|higher|lower|greater|fewer)?)\\b",
            "\\b(?:increased?|decreased?|dropped?|rose|raised?|fell|climbed?|went\\s+up|went\\s+down|gained?|lost)\\s+(?:by|to)?\\s+(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:target|goal|aim|objective|hit|reach(?:ed)?|achieve(?:d)?|attain(?:ed)?|get\\s+to|make\\s+it\\s+to)\\s+(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?)\\s+(?:steps?|calories?|hours?|minutes?|kg|pounds?|km|miles?|bpm|percent)\\b"
        ]
        
        for pattern in numberPatterns {
            if let range = text.range(of: pattern, options: .regularExpression) {
                let matchedText = String(text[range])
                let matches = try! NSRegularExpression(pattern: "(\\d+(?:\\.\\d+)?)", options: []).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                if let match = matches.first {
                    let numberRange = Range(match.range, in: matchedText)!
                    let numberString = String(matchedText[numberRange])
                    if let number = Double(numberString) {
                        return Int(number)
                    }
                }
            }
        }
        
        // Fallback to any number
        let matches = numberRegex.matches(in: text, options: [], range: NSRange(location: 0, length: text.count))
        if let match = matches.first {
            let numberRange = Range(match.range, in: text)!
            let numberString = String(text[numberRange])
            if let number = Double(numberString) {
                return Int(number)
            }
        }
        
        return nil
    }
    
    private func extractTarget(text: String) -> Int? {
        let goalPatterns = [
            "\\b(?:goal|target|aim|objective|plan|intention|aspiration|ambition|desire|want|wish|hope)\\s*(?:is|of|to|for|at)?\\s*(?:be|reach|hit|achieve|get|make|do)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:set|change|update|modify|adjust|edit|configure|make|establish|create|define|specify)\\s*(?:my|the)?\\s*(?:goal|target|aim|objective)?\\s*(?:to|at|as|for)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:reach|hit|achieve|attain|get|get\\s+to|make|do|complete|finish|accomplish|meet)\\s*(\\d+(?:\\.\\d+)?)\\s*(?:steps?|kg|kgs|kilogram|kilograms|pounds?|lbs?|km|kms|kilometer|kilometers|miles?|hours?|hrs?|minutes?|mins?|calories?|kcal|bpm)\\b",
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:steps?|kg|kgs|kilogram|kilograms|pounds?|lbs?|lb|km|kms|kilometer|kilometers|kilometre|kilometres|miles?|mi|hours?|hrs?|hr|h|minutes?|mins?|min|m|calories?|kcal|cal|cals|bpm|beats?|meter|meters|metre|metres|feet|foot|ft)\\b",
            "\\b(?:I|i)\\s+(?:want|wanna|need|must|should|have\\s+to|got\\s+to|gotta)\\s+(?:to\\s+)?(?:reach|hit|get|do|achieve|make|walk|run|burn|lose|gain|sleep)\\s*(?:to|at)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:trying|attempting|aiming|working|striving|shooting|going)\\s+(?:to|for)\\s+(?:reach|hit|get|do|achieve|make)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:daily|weekly|monthly|per\\s+day|each\\s+day|every\\s+day)\\s+(?:goal|target|aim|objective)?\\s*(?:is|of|to)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:increase|raise|boost|bump|up|improve|decrease|reduce|lower|drop|cut|bring\\s+down)\\s+(?:to|by)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:at\\s+least|minimum\\s+of|no\\s+less\\s+than|minimum|min)\\s*(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:need|needs)\\s+to\\s+(?:be|reach|hit|get)\\s*(?:at|to)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:suggest|recommend|advise|tell\\s+me|remind\\s+me|notify\\s+me).*?(?:when|if|at)\\s*(\\d+(?:\\.\\d+)?)\\b"
        ]
        
        // Try each pattern
        for pattern in goalPatterns {
            if let range = text.range(of: pattern, options: .regularExpression) {
                let matchedText = String(text[range])
                let matches = try! NSRegularExpression(pattern: "(\\d+(?:\\.\\d+)?)", options: []).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                if let match = matches.first {
                    let numberRange = Range(match.range, in: matchedText)!
                    let numberString = String(matchedText[numberRange])
                    if let number = Double(numberString) {
                        return Int(number)
                    }
                }
            }
        }
        
        // Fallback: extract any number from the text
        let matches = numberRegex.matches(in: text, options: [], range: NSRange(location: 0, length: text.count))
        if let match = matches.first {
            let numberRange = Range(match.range, in: text)!
            let numberString = String(text[numberRange])
            if let number = Double(numberString) {
                return Int(number)
            }
        }
        
        return nil
    }
    
    // MARK: - Time Normalization Helper
    
    /// Normalizes various time formats to 24-hour HH:MM format
    /// Examples:
    /// - "730" -> "07:30"
    /// - "730 pm" -> "19:30"  
    /// - "1030 pm" -> "22:30"
    /// - "2230" -> "22:30"
    /// - "7:30 am" -> "07:30"
    /// - "5 30 pm" -> "17:30"
    /// - "5.30 am" -> "05:30"
    /// - "10:30 a.m." -> "10:30"
    /// - "12:00 pm" -> "12:00"
    /// - "12:00 am" -> "00:00"
    /// Supports AM/PM formats: am, pm, a.m., p.m., a.m, p.m, a m, p m
    private func normalizeTimeFormat(_ timeString: String) -> String {
        let cleanTime = timeString.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        
        // Pattern for times like "730", "1030", "2230" (3-4 digits) without AM/PM
        if let regex = try? NSRegularExpression(pattern: "^(\\d{3,4})$"),
           let match = regex.firstMatch(in: cleanTime, range: NSRange(cleanTime.startIndex..., in: cleanTime)) {
            if let range = Range(match.range(at: 1), in: cleanTime) {
                let digits = String(cleanTime[range])
                if digits.count == 3 {
                    // e.g., "730" -> "07:30"
                    let hour = String(digits.prefix(1))
                    let minute = String(digits.suffix(2))
                    return String(format: "%02d:%@", Int(hour) ?? 0, minute)
                } else if digits.count == 4 {
                    // e.g., "2230" -> "22:30"
                    let hour = String(digits.prefix(2))
                    let minute = String(digits.suffix(2))
                    return "\(hour):\(minute)"
                }
            }
        }
        
        // Pattern for times like "730 pm", "1030 am" (digits + space + am/pm)
        // Supports: am, pm, a.m., p.m., a.m, p.m, a m, p m
        let amPmPattern = "^(\\d{1,4})\\s*(?:(?:a\\.?\\s*m\\.?)|(?:p\\.?\\s*m\\.?))$"
        if let regex = try? NSRegularExpression(pattern: amPmPattern),
           let match = regex.firstMatch(in: cleanTime, range: NSRange(cleanTime.startIndex..., in: cleanTime)) {
            if let timeRange = Range(match.range(at: 1), in: cleanTime) {
                let timeDigits = String(cleanTime[timeRange])
                let amPmText = String(cleanTime[cleanTime.index(cleanTime.startIndex, offsetBy: timeDigits.count)...]).trimmingCharacters(in: .whitespaces)
                let amPm = amPmText.starts(with: "a") ? "am" : "pm"
                
                var hour: Int = 0
                var minute: Int = 0
                
                if timeDigits.count <= 2 {
                    // e.g., "7 pm" -> hour = 7, minute = 0
                    hour = Int(timeDigits) ?? 0
                    minute = 0
                } else if timeDigits.count == 3 {
                    // e.g., "730 pm" -> hour = 7, minute = 30
                    hour = Int(String(timeDigits.prefix(1))) ?? 0
                    minute = Int(String(timeDigits.suffix(2))) ?? 0
                } else if timeDigits.count == 4 {
                    // e.g., "1030 pm" -> hour = 10, minute = 30
                    hour = Int(String(timeDigits.prefix(2))) ?? 0
                    minute = Int(String(timeDigits.suffix(2))) ?? 0
                }
                
                // Convert to 24-hour format
                if amPm == "pm" && hour != 12 {
                    hour += 12
                } else if amPm == "am" && hour == 12 {
                    hour = 0
                }
                
                return String(format: "%02d:%02d", hour, minute)
            }
        }
        
        // Pattern for "5 30 pm" or "5.30 am" format (hour [space|dot] minute am/pm)
        // Supports: am, pm, a.m., p.m., a.m, p.m, a m, p m
        let hourMinuteAmPmPattern = "^(\\d{1,2})[\\s.]+?(\\d{1,2})\\s*(?:(?:a\\.?\\s*m\\.?)|(?:p\\.?\\s*m\\.?))$"
        if let regex = try? NSRegularExpression(pattern: hourMinuteAmPmPattern),
           let match = regex.firstMatch(in: cleanTime, range: NSRange(cleanTime.startIndex..., in: cleanTime)) {
            if let hourRange = Range(match.range(at: 1), in: cleanTime),
               let minuteRange = Range(match.range(at: 2), in: cleanTime) {
                var hour = Int(cleanTime[hourRange]) ?? 0
                let minute = Int(cleanTime[minuteRange]) ?? 0
                
                // Find AM/PM indicator
                let minuteEndIndex = cleanTime.index(cleanTime.startIndex, offsetBy: NSMaxRange(match.range(at: 2)))
                let amPmText = String(cleanTime[minuteEndIndex...]).trimmingCharacters(in: .whitespaces)
                let amPm = amPmText.starts(with: "a") ? "am" : "pm"
                
                // Validate hour and minute ranges
                if hour > 12 || minute >= 60 {
                    return timeString // Return original if invalid
                }
                
                // Convert to 24-hour format
                if amPm == "pm" && hour != 12 {
                    hour += 12
                } else if amPm == "am" && hour == 12 {
                    hour = 0
                }
                
                return String(format: "%02d:%02d", hour, minute)
            }
        }
        
        // Pattern for times with colon like "7:30 pm", "10:30 am"
        // Supports: am, pm, a.m., p.m., a.m, p.m, a m, p m
        let colonAmPmPattern = "^(\\d{1,2}):(\\d{2})\\s*(?:(?:a\\.?\\s*m\\.?)|(?:p\\.?\\s*m\\.?))$"
        if let regex = try? NSRegularExpression(pattern: colonAmPmPattern),
           let match = regex.firstMatch(in: cleanTime, range: NSRange(cleanTime.startIndex..., in: cleanTime)) {
            if let hourRange = Range(match.range(at: 1), in: cleanTime),
               let minuteRange = Range(match.range(at: 2), in: cleanTime) {
                var hour = Int(cleanTime[hourRange]) ?? 0
                let minute = Int(cleanTime[minuteRange]) ?? 0
                
                // Find AM/PM indicator
                let minuteEndIndex = cleanTime.index(cleanTime.startIndex, offsetBy: NSMaxRange(match.range(at: 2)))
                let amPmText = String(cleanTime[minuteEndIndex...]).trimmingCharacters(in: .whitespaces)
                let amPm = amPmText.starts(with: "a") ? "am" : "pm"
                
                // Convert to 24-hour format
                if amPm == "pm" && hour != 12 {
                    hour += 12
                } else if amPm == "am" && hour == 12 {
                    hour = 0
                }
                
                return String(format: "%02d:%02d", hour, minute)
            }
        }
        
        // Pattern for 24-hour format like "22:30", "07:30"
        let twentyFourHourPattern = "^(\\d{1,2}):(\\d{2})$"
        if let regex = try? NSRegularExpression(pattern: twentyFourHourPattern),
           let match = regex.firstMatch(in: cleanTime, range: NSRange(cleanTime.startIndex..., in: cleanTime)) {
            if let hourRange = Range(match.range(at: 1), in: cleanTime),
               let minuteRange = Range(match.range(at: 2), in: cleanTime) {
                let hour = Int(cleanTime[hourRange]) ?? 0
                let minute = Int(cleanTime[minuteRange]) ?? 0
                return String(format: "%02d:%02d", hour, minute)
            }
        }
        
        // Return original string if no pattern matches
        return timeString
    }
    
    // MARK: - Timer Duration Normalization Helper
    
    /// Normalizes timer duration to seconds
    /// Examples:
    /// - "5 min" -> 300 (seconds)
    /// - "2 hr" -> 7200 (seconds)
    /// - "1 hour 30 minutes" -> 5400 (seconds)
    /// - "45 sec" -> 45 (seconds)
    /// - "1:30" -> 90 (seconds, interpreted as MM:SS)
    /// - "2:30:45" -> 9045 (seconds, HH:MM:SS)
    private func normalizeTimerDuration(_ text: String, matchedValue: String) -> Int {
        let cleanText = text.lowercased()
        
        // Check for combined duration patterns first
        
        // "1 hour 30 minutes 45 seconds" pattern
        if let regex = try? NSRegularExpression(pattern: "(\\d+)\\s*(?:hours?|hrs?|hr|h)\\s*(?:and\\s+)?(\\d+)\\s*(?:minutes?|mins?|min|m)\\s*(?:and\\s+)?(\\d+)\\s*(?:seconds?|secs?|sec|s)", options: [.caseInsensitive]),
           let match = regex.firstMatch(in: cleanText, range: NSRange(cleanText.startIndex..., in: cleanText)) {
            if let hoursRange = Range(match.range(at: 1), in: cleanText),
               let minutesRange = Range(match.range(at: 2), in: cleanText),
               let secondsRange = Range(match.range(at: 3), in: cleanText) {
                let hours = Int(cleanText[hoursRange]) ?? 0
                let minutes = Int(cleanText[minutesRange]) ?? 0
                let seconds = Int(cleanText[secondsRange]) ?? 0
                return hours * 3600 + minutes * 60 + seconds
            }
        }
        
        // "1 hour 30 minutes" or "2h 45m" pattern
        if let regex = try? NSRegularExpression(pattern: "(\\d+)\\s*(?:hours?|hrs?|hr|h)\\s*(?:and\\s+)?(\\d+)\\s*(?:minutes?|mins?|min|m)", options: [.caseInsensitive]),
           let match = regex.firstMatch(in: cleanText, range: NSRange(cleanText.startIndex..., in: cleanText)) {
            if let hoursRange = Range(match.range(at: 1), in: cleanText),
               let minutesRange = Range(match.range(at: 2), in: cleanText) {
                let hours = Int(cleanText[hoursRange]) ?? 0
                let minutes = Int(cleanText[minutesRange]) ?? 0
                return hours * 3600 + minutes * 60
            }
        }
        
        // "30 minutes 45 seconds" or "30m 45s" pattern
        if let regex = try? NSRegularExpression(pattern: "(\\d+)\\s*(?:minutes?|mins?|min|m)\\s*(?:and\\s+)?(\\d+)\\s*(?:seconds?|secs?|sec|s)", options: [.caseInsensitive]),
           let match = regex.firstMatch(in: cleanText, range: NSRange(cleanText.startIndex..., in: cleanText)) {
            if let minutesRange = Range(match.range(at: 1), in: cleanText),
               let secondsRange = Range(match.range(at: 2), in: cleanText) {
                let minutes = Int(cleanText[minutesRange]) ?? 0
                let seconds = Int(cleanText[secondsRange]) ?? 0
                return minutes * 60 + seconds
            }
        }
        
        // Check for time format patterns (MM:SS or HH:MM:SS)
        
        // HH:MM:SS format
        if let regex = try? NSRegularExpression(pattern: "^(\\d+):(\\d+):(\\d+)$"),
           let match = regex.firstMatch(in: matchedValue.trimmingCharacters(in: .whitespaces), range: NSRange(matchedValue.startIndex..., in: matchedValue)) {
            if let hoursRange = Range(match.range(at: 1), in: matchedValue),
               let minutesRange = Range(match.range(at: 2), in: matchedValue),
               let secondsRange = Range(match.range(at: 3), in: matchedValue) {
                let hours = Int(matchedValue[hoursRange]) ?? 0
                let minutes = Int(matchedValue[minutesRange]) ?? 0
                let seconds = Int(matchedValue[secondsRange]) ?? 0
                return hours * 3600 + minutes * 60 + seconds
            }
        }
        
        // MM:SS format (assuming minutes:seconds for timer)
        if let regex = try? NSRegularExpression(pattern: "^(\\d+):(\\d+)$"),
           let match = regex.firstMatch(in: matchedValue.trimmingCharacters(in: .whitespaces), range: NSRange(matchedValue.startIndex..., in: matchedValue)) {
            if let minutesRange = Range(match.range(at: 1), in: matchedValue),
               let secondsRange = Range(match.range(at: 2), in: matchedValue) {
                let minutes = Int(matchedValue[minutesRange]) ?? 0
                let seconds = Int(matchedValue[secondsRange]) ?? 0
                return minutes * 60 + seconds
            }
        }
        
        // Check for single unit patterns
        let cleanedValue = matchedValue.replacingOccurrences(of: "[^\\d.]", with: "", options: .regularExpression)
        guard let value = Double(cleanedValue) else { return 0 }
        
        if cleanText.range(of: "\\b(?:hours?|hrs?|hr|h)\\b", options: .regularExpression) != nil {
            return Int(value * 3600)
        } else if cleanText.range(of: "\\b(?:minutes?|mins?|min|m)\\b", options: .regularExpression) != nil {
            return Int(value * 60)
        } else if cleanText.range(of: "\\b(?:seconds?|secs?|sec|s)\\b", options: .regularExpression) != nil {
            return Int(value)
        } else {
            return Int(value) // Default to treating as seconds if no unit specified
        }
    }
    
    
    private func extractValue(text: String, intent: String) -> Any? {
        switch intent {
        case "LogEvent":
            // Weight patterns
            let weightPatterns = [
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|k\\.?g\\.?)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:pounds?|lbs?|lb|pound\\s+weight|#|lbs\\s+weight|lb\\s+weight)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:grams?|grammes?|g|gm|gms|g\\.?m?\\.?s?\\.?)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:stone|ounces?|ounce|oz)\\b",
                "\\b(?:weigh|weight|weighing|weighed|mass|body\\s+weight|body\\s+mass|scale|heavy|light).*?(\\d+(?:\\.\\d+)?)\\b",
                "\\b(\\d+(?:\\.\\d+)?).*?(?:weigh|weight|weighing|weighed|mass|body\\s+weight|body\\s+mass|scale|heavy|light)\\b",
                "\\b(?:i\\s+)?(?:am|was|have|had|got|get|getting|became|become|becoming|turned|turn|turning|went|go|going|reached|reach|reaching|hit|hitting|measured|measure|measuring|recorded|record|recording|logged|log|logging|entered|enter|entering|input|inputted|inputting|typed|type|typing|added|add|adding).*?(\\d+(?:\\.\\d+)?)\\s*(?:kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|k\\.?g\\.?|pounds?|lbs?|lb|pound\\s+weight|#|lbs\\s+weight|lb\\s+weight|grams?|grammes?|g|gm|gms|g\\.?m?\\.?s?\\.?|stone|ounces?|ounce|oz)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|k\\.?g\\.?|pounds?|lbs?|lb|pound\\s+weight|#|lbs\\s+weight|lb\\s+weight|grams?|grammes?|g|gm|gms|g\\.?m?\\.?s?\\.?|stone|ounces?|ounce|oz).*?(?:i\\s+)?(?:am|was|have|had|got|get|getting|became|become|becoming|turned|turn|turning|went|go|going|reached|reach|reaching|hit|hitting|measured|measure|measuring|recorded|record|recording|logged|log|logging|entered|enter|entering|input|inputted|inputting|typed|type|typing|added|add|adding)\\b",
                "\\b(?:my|current|new|latest|recent|today's|this\\s+morning's|yesterday's|last\\s+night's)\\s*(?:weight|mass|body\\s+weight|body\\s+mass).*?(\\d+(?:\\.\\d+)?)\\b",
                "\\b(\\d+(?:\\.\\d+)?).*?(?:my|current|new|latest|recent|today's|this\\s+morning's|yesterday's|last\\s+night's)\\s*(?:weight|mass|body\\s+weight|body\\s+mass)\\b",
                "\\b(?:scale|scales|weighing\\s+scale|bathroom\\s+scale|digital\\s+scale|weight\\s+scale).*?(?:shows?|showed|showing|says?|said|saying|reads?|read|reading|displays?|displayed|displaying|indicates?|indicated|indicating|reports?|reported|reporting|gives?|gave|giving|tells?|told|telling).*?(\\d+(?:\\.\\d+)?)\\b",
                "\\b(\\d+(?:\\.\\d+)?).*?(?:scale|scales|weighing\\s+scale|bathroom\\s+scale|digital\\s+scale|weight\\s+scale).*?(?:shows?|showed|showing|says?|said|saying|reads?|read|reading|displays?|displayed|displaying|indicates?|indicated|indicating|reports?|reported|reporting|gives?|gave|giving|tells?|told|telling)\\b"
            ]
            
            for pattern in weightPatterns {
                if let range = text.range(of: pattern, options: .regularExpression) {
                    let matchedText = String(text[range])
                    let matches = try! NSRegularExpression(pattern: "(\\d+(?:\\.\\d+)?)", options: []).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                    if let match = matches.first {
                        let numberRange = Range(match.range, in: matchedText)!
                        let numberString = String(matchedText[numberRange])
                        if let number = Double(numberString) {
                            return number
                        }
                    }
                }
            }
            
            // General number extraction for other log events
            let generalPatterns = [
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:ml|milliliters?|millilitres?|l|liters?|litres?|ltr|ltrs|l\\.?t?r?\\.?s?\\.?|cups?|glasses?|bottles?|fl\\s+oz|fluid\\s+ounces?|pints?|quarts?|gallons?)\\b",
                "\\b(?:drank|drink|drinking|drunk|consumed|consume|consuming|had|have|having|took|take|taking|ingested|ingest|ingesting|swallowed|swallow|swallowing).*?(\\d+(?:\\.\\d+)?)\\b",
                "\\b(\\d+(?:\\.\\d+)?).*?(?:drank|drink|drinking|drunk|consumed|consume|consuming|had|have|having|took|take|taking|ingested|ingest|ingesting|swallowed|swallow|swallowing)\\b",
                "\\b(?:amount|quantity|volume|dose|serving|portion|measurement|count|number|total|sum).*?(\\d+(?:\\.\\d+)?)\\b",
                "\\b(\\d+(?:\\.\\d+)?).*?(?:amount|quantity|volume|dose|serving|portion|measurement|count|number|total|sum)\\b"
            ]
            
            for pattern in generalPatterns {
                if let range = text.range(of: pattern, options: .regularExpression) {
                    let matchedText = String(text[range])
                    let matches = try! NSRegularExpression(pattern: "(\\d+(?:\\.\\d+)?)", options: []).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                    if let match = matches.first {
                        let numberRange = Range(match.range, in: matchedText)!
                        let numberString = String(matchedText[numberRange])
                        if let number = Double(numberString) {
                            return number
                        }
                    }
                }
            }
            
        case "TimerStopwatch":
            // Determine if this is a timer (duration) or alarm (specific time) context
            let isTimerContext = text.range(of: "\\b(?:timer|stopwatch|countdown|count\\s+down)\\b", options: .regularExpression) != nil
            let isAlarmContext = text.range(of: "\\b(?:alarm|wake|remind|alert)\\b", options: .regularExpression) != nil
            
            let timePatterns = [
                // Space/dot separated time with AM/PM - highest priority for "5 30 pm", "10.30 am" format
                "\\b(\\d{1,2}[\\s.]+?\\d{1,2}\\s*(?:(?:a\\.?\\s*m\\.?)|(?:p\\.?\\s*m\\.?)))\\b",
                
                // Time with AM/PM - capture full time including AM/PM
                "\\b(\\d{1,2}(?::\\d{2})?\\s*(?:(?:a\\.?\\s*m\\.?)|(?:p\\.?\\s*m\\.?)))\\b",
                "\\b(\\d{3,4}\\s*(?:(?:a\\.?\\s*m\\.?)|(?:p\\.?\\s*m\\.?)))\\b", // For "1030 pm" format
                
                // Duration patterns for timers
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:hours?|hrs?|hr|h)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:minutes?|mins?|min|m)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:seconds?|secs?|sec|s)\\b",
                
                // Combined time patterns
                "\\b(\\d+)\\s*(?:h|hr|hours?)\\s*(?:and\\s+)?(\\d+)\\s*(?:m|min|minutes?)\\b",
                "\\b(\\d+)\\s*(?:m|min|minutes?)\\s*(?:and\\s+)?(\\d+)\\s*(?:s|sec|seconds?)\\b",
                "\\b(\\d+):(\\d+):(\\d+)\\b", // HH:MM:SS format
                "\\b(\\d+):(\\d+)\\b", // MM:SS or HH:MM format
                
                // Duration keywords
                "\\b(?:for|during|lasting|takes?)\\s+(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|hours?|hrs?|seconds?|secs?)\\b",
                
                // Timer-specific patterns
                "\\b(?:set|start|begin|run|timer|stopwatch)\\s*(?:for|to|at)?\\s*(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|hours?|hrs?|seconds?|secs?)\\b",
                
                // Alarm-specific patterns - capture full time with AM/PM (prioritize space-separated)
                "\\b(?:alarm|wake|remind|alert)\\s*(?:at|for|in)?\\s*(\\d{1,2}[\\s.]+?\\d{1,2}\\s*(?:(?:a\\.?\\s*m\\.?)|(?:p\\.?\\s*m\\.?)))\\b",
                "\\b(?:alarm|wake|remind|alert)\\s*(?:at|for|in)?\\s*(\\d{1,2}(?::\\d{2})?\\s*(?:(?:a\\.?\\s*m\\.?)|(?:p\\.?\\s*m\\.?)))\\b",
                "\\b(?:alarm|wake|remind|alert)\\s*(?:at|for|in)?\\s*(\\d{3,4}\\s*(?:(?:a\\.?\\s*m\\.?)|(?:p\\.?\\s*m\\.?)))\\b",
                
                // "In X time" patterns
                "\\b(?:in|after|within)\\s+(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|hours?|hrs?|seconds?|secs?)\\b",
                
                // Countdown patterns
                "\\b(?:countdown|count\\s+down)\\s*(?:from|for)?\\s*(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|hours?|hrs?|seconds?|secs?)\\b",
                
                // Direct time patterns for alarm setting - capture full time with AM/PM (prioritize space-separated)
                "\\b(?:set|create|make).*?(?:alarm|wake).*?(?:for|at)\\s*(\\d{1,2}[\\s.]+?\\d{1,2}\\s*(?:am|pm))\\b",
                "\\b(?:set|create|make).*?(?:alarm|wake).*?(?:for|at)\\s*(\\d{3,4}\\s*(?:am|pm))\\b",
                "\\b(?:set|create|make).*?(?:alarm|wake).*?(?:for|at)\\s*(\\d{1,2}(?::\\d{2})?\\s*(?:am|pm))\\b",
                "\\b(?:alarm|wake).*?(?:for|at)\\s*(\\d{1,2}[\\s.]+?\\d{1,2}\\s*(?:am|pm))\\b",
                "\\b(?:alarm|wake).*?(?:for|at)\\s*(\\d{3,4}\\s*(?:am|pm))\\b",
                "\\b(?:alarm|wake).*?(?:for|at)\\s*(\\d{1,2}(?::\\d{2})?\\s*(?:am|pm))\\b",
                
                // Standalone numeric patterns for alarm context (no AM/PM)
                "\\b(?:alarm|wake).*?(?:for|at)\\s*(\\d{3,4})\\b",
                "\\b(?:alarm|wake).*?(?:for|at)\\s*(\\d{1,2}(?::\\d{2})?)\\b"
            ]
            
            for pattern in timePatterns {
                if let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]),
                   let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) {
                    
                    // Handle combined time formats for duration (e.g., "1h 30m" or "1:30")
                    if match.numberOfRanges > 2,
                       let range2 = Range(match.range(at: 2), in: text),
                       !String(text[range2]).isEmpty,
                       text.range(of: "\\b(?:am|pm)\\b", options: .regularExpression) == nil {
                        
                        if isTimerContext {
                            // For timers, convert combined duration to seconds
                            if let matchRange = Range(match.range, in: text) {
                                return normalizeTimerDuration(text, matchedValue: String(text[matchRange]))
                            }
                        } else {
                            // For other contexts, return total minutes as before
                            if let range1 = Range(match.range(at: 1), in: text) {
                                let hours = Int(text[range1]) ?? 0
                                let minutes = Int(text[range2]) ?? 0
                                return String(hours * 60 + minutes)
                            }
                        }
                    }
                    
                    if let timeRange = Range(match.range(at: 1), in: text) {
                        let timeValue = String(text[timeRange])
                        
                        // Check if this is a time format (contains AM/PM or looks like time in alarm context)
                        let containsAmPm = timeValue.range(of: "\\b(?:(?:a\\.?\\s*m\\.?)|(?:p\\.?\\s*m\\.?))\\b", options: .regularExpression) != nil
                        let containsSpaceOrDot = timeValue.range(of: "[\\s.]\\d", options: .regularExpression) != nil
                        let looksLikeTime = timeValue.range(of: "^\\d{3,4}$", options: .regularExpression) != nil || timeValue.contains(":") || containsSpaceOrDot
                        
                        if containsAmPm || (isAlarmContext && looksLikeTime) {
                            // This is an alarm time, normalize to HH:MM format
                            return normalizeTimeFormat(timeValue)
                        } else if isTimerContext {
                            // This is a timer duration, convert to seconds
                            if let matchRange = Range(match.range, in: text) {
                                return normalizeTimerDuration(text, matchedValue: String(text[matchRange]))
                            }
                        }
                        
                        // Return the time value for other cases
                        return timeValue
                    }
                }
            }
            
            // Fallback to any number
            let matches = numberSequenceRegex.matches(in: text, options: [], range: NSRange(location: 0, length: text.count))
            if let match = matches.first,
               let numberRange = Range(match.range(at: 1), in: text) {
                if let number = Int(text[numberRange]) {
                    return number
                }
            }
            
        default:
            // General number extraction for other intents
            let generalPatterns = [
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:units?|items?|pieces?|counts?|amounts?|quantities?|numbers?|values?|figures?|measurements?|readings?|totals?|sums?)\\b",
                "\\b(?:value|number|amount|quantity|count|total|sum|figure|measurement|reading|score|rating|level|degree|percentage|percent|rate|ratio|proportion).*?(\\d+(?:\\.\\d+)?)\\b",
                "\\b(\\d+(?:\\.\\d+)?).*?(?:value|number|amount|quantity|count|total|sum|figure|measurement|reading|score|rating|level|degree|percentage|percent|rate|ratio|proportion)\\b"
            ]
            
            for pattern in generalPatterns {
                if let range = text.range(of: pattern, options: .regularExpression) {
                    let matchedText = String(text[range])
                    let matches = try! NSRegularExpression(pattern: "(\\d+(?:\\.\\d+)?)", options: []).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                    if let match = matches.first {
                        let numberRange = Range(match.range, in: matchedText)!
                        let numberString = String(matchedText[numberRange])
                        if let number = Double(numberString) {
                            return Int(number)
                        }
                    }
                }
            }
            
            // Ultimate fallback - any number
            let matches = numberRegex.matches(in: text, options: [], range: NSRange(location: 0, length: text.count))
            if let match = matches.first {
                let numberRange = Range(match.range, in: text)!
                let numberString = String(text[numberRange])
                if let number = Double(numberString) {
                    return Int(number)
                }
            }
        }
        
        return nil
    }
    
    
    private func extractFeature(text: String) -> String? {
        let features: [String: String] = [
            "do not disturb": "\\b(?:do\\s+not\\s+disturb|dnd|d\\.?n\\.?d\\.?|silent\\s+mode|silence|quiet\\s+mode|mute|muted|no\\s+disturb|don'?t\\s+disturb|silence\\s+notifications?|quiet\\s+hours?|sleep\\s+mode|bedtime\\s+mode|focus\\s+mode|zen\\s+mode|peaceful\\s+mode|undisturbed|interruption\\s+free|notification\\s+silence)\\b",
            "AOD": "\\b(?:AOD|aod|a\\.?o\\.?d\\.?|always\\s+on\\s+display|always-on\\s+display|always\\s+on|screen\\s+always\\s+on|display\\s+always\\s+on|persistent\\s+display|constant\\s+display|continuous\\s+display|keep\\s+screen\\s+on|screen\\s+stays\\s+on|display\\s+on|ambient\\s+display|glance\\s+screen|standby\\s+screen)\\b",
            "raise to wake": "\\b(?:raise\\s+to\\s+wake|lift\\s+to\\s+wake|tap\\s+to\\s+wake|double\\s+tap\\s+to\\s+wake|touch\\s+to\\s+wake|wrist\\s+raise|raise\\s+wrist|lift\\s+wrist|wake\\s+on\\s+raise|wake\\s+on\\s+lift|wake\\s+on\\s+tap|wake\\s+on\\s+touch|pick\\s+up\\s+to\\s+wake|gesture\\s+wake|motion\\s+wake|tilt\\s+to\\s+wake|wake\\s+gesture|screen\\s+wake|auto\\s+wake|smart\\s+wake)\\b",
            "vibration": "\\b(?:vibration|vibrate|vibrating|haptic|haptics|buzz|buzzing|rumble|rumbling|tactile|tactile\\s+feedback|vibration\\s+feedback|motor|vibration\\s+motor|shake|shaking|pulse|pulsing|vibe|vibes|vibrate\\s+mode|silent\\s+vibrate|ring\\s+vibrate)\\b",
            "volume": "\\b(?:volume|sound\\s+level|sound\\s+volume|audio\\s+level|audio\\s+volume|loudness|loud|quiet|soft|sound|audio|speaker\\s+volume|media\\s+volume|ringtone\\s+volume|notification\\s+volume|alarm\\s+volume|call\\s+volume|ringer|sound\\s+output|audio\\s+output|volume\\s+level)\\b",
            "torch": "\\b(?:torch|flashlight|flash\\s+light|led\\s+light|led\\s+torch|camera\\s+flash|light|lamp|lantern|beam|illumination|bright\\s+light|phone\\s+light|mobile\\s+light|emergency\\s+light|torch\\s+light|strobe|strobe\\s+light|spotlight|searchlight|headlight|flash\\s+lamp|portable\\s+light|hand\\s+light|led\\s+flash|camera\\s+light|phone\\s+torch|device\\s+light|built-in\\s+light|integrated\\s+light)\\b"
        ]
        // Sort by pattern length for more specific matching
        let sortedFeatures = features.sorted { $0.value.count > $1.value.count }
        for (feature, pattern) in sortedFeatures {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return feature
            }
        }
        return nil
    }
    
    
    private func extractState(text: String) -> String? {
        if text.range(of: "\\b(?:turn\\s+on|enable|enabled|enabling|activate|activated|activating|switch\\s+on|start|started|starting|power\\s+on|boot|boot\\s+up|fire\\s+up|launch|open|unmute|unmuted|resume|allow|permit|engage|engaged|engaging|set\\s+on|put\\s+on|make\\s+it\\s+on|get\\s+it\\s+on|bring\\s+up|wake\\s+up|light\\s+up|flip\\s+on|on)\\b", options: [.regularExpression, .caseInsensitive]) != nil {
            return "on"
        }
        
        if text.range(of: "\\b(?:turn\\s+off|disable|disabled|disabling|deactivate|deactivated|deactivating|switch\\s+off|stop|stopped|stopping|shut\\s+off|shut\\s+down|power\\s+off|kill|close|mute|muted|pause|paused|block|deny|disengage|disengaged|disengaging|set\\s+off|put\\s+off|make\\s+it\\s+off|get\\s+it\\s+off|bring\\s+down|sleep|suspend|flip\\s+off|cut\\s+off|off)\\b", options: [.regularExpression, .caseInsensitive]) != nil {
            return "off"
        }
        
        if text.range(of: "\\b(?:increase|increased|increasing|up|higher|raise|raised|raising|boost|boosted|boosting|amplify|amplified|amplifying|enhance|enhanced|enhancing|elevate|elevated|elevating|pump\\s+up|turn\\s+up|crank\\s+up|ramp\\s+up|scale\\s+up|step\\s+up|jack\\s+up|bump\\s+up|push\\s+up|bring\\s+up|make\\s+it\\s+higher|louder|brighter|stronger|more|maximize|max\\s+out|intensify)\\b", options: [.regularExpression, .caseInsensitive]) != nil {
            return "increase"
        }
        
        if text.range(of: "\\b(?:decrease|decreased|decreasing|down|lower|lowered|lowering|reduce|reduced|reducing|diminish|diminished|diminishing|lessen|lessened|lessening|drop|dropped|dropping|cut|cutting|turn\\s+down|bring\\s+down|scale\\s+down|step\\s+down|tone\\s+down|dial\\s+down|wind\\s+down|ramp\\s+down|make\\s+it\\s+lower|quieter|dimmer|weaker|less|minimize|min\\s+out|soften)\\b", options: [.regularExpression, .caseInsensitive]) != nil {
            return "decrease"
        }
        
        return nil
    }
    
    private func extractAction(text: String) -> String? {
        let actions: [String: String] = [
            "set": "\\b(?:set|setup|set\\s+up|configure|configuration|adjust|adjustment|change|modify|edit|customize|establish|define|specify|determine|fix|assign|allocate|program|preset|input|enter|put\\s+in|make\\s+it|arrange|organize|prepare)\\b",
            
            "start": "\\b(?:start|started|starting|begin|began|beginning|initiate|initiated|initiating|launch|launched|launching|commence|commencing|kick\\s+off|fire\\s+up|boot\\s+up|power\\s+on|turn\\s+on|switch\\s+on|activate|enable|engage|trigger|run|execute|go|let'?s\\s+go|get\\s+going|get\\s+started)\\b",
            
            "stop": "\\b(?:stop|stopped|stopping|end|ended|ending|finish|finished|finishing|terminate|terminated|terminating|cease|halt|pause|paused|pausing|kill|abort|cancel|cancelled|canceling|quit|exit|close|shut\\s+down|power\\s+off|turn\\s+off|switch\\s+off|deactivate|disable|disengage|cut\\s+off)\\b",
            
            "call": "\\b(?:call|calling|phone|dial|dialing|ring|ringing|contact|reach|reach\\s+out|get\\s+in\\s+touch|give\\s+a\\s+call|make\\s+a\\s+call|place\\s+a\\s+call|telephone|buzz|video\\s+call|voice\\s+call|facetime)\\b",
            
            "message": "\\b(?:message|messaging|text|texting|sms|send|sending|sent|write|compose|type|drop\\s+a\\s+message|send\\s+a\\s+text|shoot\\s+a\\s+message|ping|dm|direct\\s+message|whatsapp|imessage|chat|msg)\\b",
            
            "open": "\\b(?:open|opened|opening|launch|launched|launching|start|show|display|view|access|load|bring\\s+up|pull\\s+up|fire\\s+up|boot|go\\s+to|navigate\\s+to|switch\\s+to|take\\s+me\\s+to)\\b",
            
            "check": "\\b(?:check|checking|verify|verifying|examine|look|looking|see|review|inspect|assess|evaluate|monitor|watch|observe|scan|browse|view|find\\s+out|tell\\s+me|show\\s+me|let\\s+me\\s+see|give\\s+me|what'?s|how'?s|any)\\b",
            
            "measure": "\\b(?:measure|measuring|measured|test|testing|tested|record|recording|recorded|track|tracking|log|logging|logged|take|capture|monitor|scan|read|reading|sample|collect|gauge|assess|evaluate)\\b",

            "play": "\\b(?:play|playing|played|resume|resuming|resumed|continue|continuing|continued|unpause|unpausing|unpaused|start\\s+playing|begin\\s+playing|kick\\s+off|fire\\s+up|roll|rolling|spun|spin|spinning)\\b",

            "pause": "\\b(?:pause|pausing|paused|hold|holding|held|freeze|freezing|frozen|stop\\s+temporarily|suspend|suspended|suspending|halt\\s+temporarily|break|breaking|broke|interrupt|interrupting|interrupted)\\b",

            "increase": "\\b(?:increase|increased|increasing|up|higher|raise|raised|raising|boost|boosted|boosting|amplify|amplified|amplifying|enhance|enhanced|enhancing|elevate|elevated|elevating|pump\\s+up|turn\\s+up|crank\\s+up|ramp\\s+up|scale\\s+up|step\\s+up|jack\\s+up|bump\\s+up|push\\s+up|bring\\s+up|make\\s+it\\s+higher|louder|brighter|stronger|more|maximize|max\\s+out|intensify)\\b",

            "decrease": "\\b(?:decrease|decreased|decreasing|down|lower|lowered|lowering|reduce|reduced|reducing|diminish|diminished|diminishing|lessen|lessened|lessening|drop|dropped|dropping|cut|cutting|turn\\s+down|bring\\s+down|scale\\s+down|step\\s+down|tone\\s+down|dial\\s+down|wind\\s+down|ramp\\s+down|make\\s+it\\s+lower|quieter|dimmer|weaker|less|minimize|min\\s+out|soften)\\b",

            "skip_next": "\\b(?:skip\\s+(?:forward|next|ahead)|next\\s+(?:track|song|chapter|episode|video|clip)|forward\\s+(?:to\\s+next|one)|advance\\s+(?:to\\s+next|one)|go\\s+(?:to\\s+next|forward\\s+one)|jump\\s+(?:to\\s+next|forward)|fast\\s+forward\\s+(?:to\\s+next|one)|move\\s+(?:to\\s+next|forward)|switch\\s+(?:to\\s+next|forward))\\b",

            "skip_previous": "\\b(?:skip\\s+(?:back|previous|backward)|previous\\s+(?:track|song|chapter|episode|video|clip)|back\\s+(?:to\\s+previous|one)|go\\s+(?:to\\s+previous|back\\s+one)|jump\\s+(?:to\\s+previous|back)|rewind\\s+(?:to\\s+previous|one)|move\\s+(?:to\\s+previous|back)|switch\\s+(?:to\\s+previous|back)|restart\\s+(?:track|song|current))\\b",

            "fast_forward": "\\b(?:fast\\s+forward|speed\\s+up|forward\\s+(?:quickly|fast)|accelerate\\s+(?:playback|forward)|rush\\s+forward|zoom\\s+forward|hurry\\s+forward|quick\\s+forward|rapid\\s+forward|expedite\\s+forward|double\\s+speed|triple\\s+speed|increase\\s+speed)\\b",

            "rewind": "\\b(?:rewind|rewinding|rewound|fast\\s+backward|reverse\\s+(?:quickly|fast)|back\\s+up|go\\s+back|reverse\\s+playback|backward\\s+(?:quickly|fast)|retreat|retreating|retreated|regress|regressing|regressed|slow\\s+reverse|reverse\\s+slowly)\\b",

            "seek": "\\b(?:seek|seeking|sought|jump\\s+to|go\\s+to|move\\s+to|navigate\\s+to|position\\s+to|scrub\\s+to|advance\\s+to|retreat\\s+to|set\\s+position|change\\s+position|adjust\\s+position|locate\\s+to|find\\s+position|progress\\s+to)\\b",

            "mute": "\\b(?:mute|muting|muted|silence|silencing|silenced|quiet|quieting|quieted|turn\\s+off\\s+sound|disable\\s+sound|kill\\s+sound|cut\\s+sound|no\\s+sound|sound\\s+off|audio\\s+off|volume\\s+off|silent\\s+mode)\\b",

            "unmute": "\\b(?:unmute|unmuting|unmuted|unsilence|unsilencing|unsilenced|unquiet|unquieting|unquieted|turn\\s+on\\s+sound|enable\\s+sound|restore\\s+sound|bring\\s+back\\s+sound|sound\\s+on|audio\\s+on|volume\\s+on|exit\\s+silent\\s+mode)\\b",

            "fullscreen": "\\b(?:full\\s+screen|fullscreen|full-screen|maximize|maximizing|maximized|expand|expanding|expanded|enlarge|enlarging|enlarged|stretch|stretching|stretched|fill\\s+screen|screen\\s+fill|wide\\s+screen|cinema\\s+mode|theater\\s+mode|immersive\\s+mode)\\b",

            "captions": "\\b(?:caption|captions|subtitle|subtitles|closed\\s+caption|closed\\s+captions|cc|cc'?s|text|texts|transcript|transcripts|sub|subs|overlay|overlays|dialogue|dialogues|speech\\s+text|spoken\\s+text|audio\\s+description)\\b",

            "speed": "\\b(?:speed|speeding|sped|playback\\s+speed|play\\s+speed|rate|rates|pace|pacing|paced|tempo|tempos|tempoing|tempoed|rhythm|rhythms|rhythmical|velocity|velocities|quickness|quicknesses|rapidity|rapidities)\\b",

            "shuffle": "\\b(?:shuffle|shuffling|shuffled|random|randomize|randomizing|randomized|mix|mixing|mixed|scramble|scrambling|scrambled|jumble|jumbling|jumbled|disorder|disordering|disordered|rearrange|rearranging|rearranged)\\b",

            "repeat": "\\b(?:repeat|repeating|repeated|loop|looping|looped|cycle|cycling|cycled|replay|replaying|replayed|encore|encoring|encored|again|repeat\\s+mode|loop\\s+mode|continuous\\s+play|infinite\\s+play)\\b"
        ]
        
        // Sort actions by pattern length for more specific matching
        let sortedActions = actions.sorted { $0.value.count > $1.value.count }
        
        for (action, pattern) in sortedActions {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return action
            }
        }
        
        return nil
    }
    
    
    private func extractTimerAction(text: String) -> String? {
        let timerActions: [String: String] = [
            "set": "\\b(?:set|setup|set\\s+up|configure|configuration|adjust|adjustment|change|modify|edit|customize|establish|define|specify|determine|fix|assign|allocate|program|preset|input|enter|put\\s+in|make\\s+it|arrange|organize|prepare|setting)\\b",
            
            "start": "\\b(?:start|started|starting|begin|began|beginning|initiate|initiated|initiating|launch|launched|launching|commence|commencing|kick\\s+off|fire\\s+up|boot\\s+up|power\\s+on|turn\\s+on|switch\\s+on|activate|enable|engage|trigger|run|execute|go|let'?s\\s+go|get\\s+going|get\\s+started)\\b",
            
            "stop": "\\b(?:stop|stopped|stopping|end|ended|ending|finish|finished|finishing|terminate|terminated|terminating|cease|halt|pause|paused|pausing|kill|abort|cancel|cancelled|canceling|quit|exit|close|shut\\s+down|power\\s+off|turn\\s+off|switch\\s+off|deactivate|disable|disengage|cut\\s+off)\\b"
        ]
        
        // Sort actions by pattern specificity (longer patterns first for better matching)
        let sortedActions = timerActions.sorted { $0.value.count > $1.value.count }
        
        for (action, pattern) in sortedActions {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return action
            }
        }
        
        return nil
    }
    
    private func extractMediaAction(text: String) -> String? {
        let mediaActions: [String: String] = [
            "play": "\\b(?:play|playing|played|resume|resuming|resumed|continue|continuing|continued|unpause|unpausing|unpaused|start\\s+playing|begin\\s+playing|kick\\s+off|fire\\s+up|roll|rolling|spun|spin|spinning)\\b",
            
            "pause": "\\b(?:pause|pausing|paused|hold|holding|held|freeze|freezing|frozen|stop\\s+temporarily|suspend|suspended|suspending|halt\\s+temporarily|break|breaking|broke|interrupt|interrupting|interrupted)\\b",
            
            "stop": "\\b(?:stop|stopped|stopping|end|ended|ending|finish|finished|finishing|terminate|terminated|terminating|cease|halt|kill|abort|cancel|cancelled|canceling|quit|exit|close|shut\\s+down|power\\s+off|turn\\s+off|switch\\s+off|deactivate|disable|disengage|cut\\s+off)\\b",
            
            "skip_next": "\\b(?:skip\\s+(?:forward|next|ahead)|next\\s+(?:track|song|chapter|episode|video|clip)|forward\\s+(?:to\\s+next|one)|advance\\s+(?:to\\s+next|one)|go\\s+(?:to\\s+next|forward\\s+one)|jump\\s+(?:to\\s+next|forward)|fast\\s+forward\\s+(?:to\\s+next|one)|move\\s+(?:to\\s+next|forward)|switch\\s+(?:to\\s+next|forward))\\b",
            
            "skip_previous": "\\b(?:skip\\s+(?:back|previous|backward)|previous\\s+(?:track|song|chapter|episode|video|clip)|back\\s+(?:to\\s+previous|one)|go\\s+(?:to\\s+previous|back\\s+one)|jump\\s+(?:to\\s+previous|back)|rewind\\s+(?:to\\s+previous|one)|move\\s+(?:to\\s+previous|back)|switch\\s+(?:to\\s+previous|back)|restart\\s+(?:track|song|current))\\b",
            
            "fast_forward": "\\b(?:fast\\s+forward|speed\\s+up|forward\\s+(?:quickly|fast)|accelerate\\s+(?:playback|forward)|rush\\s+forward|zoom\\s+forward|hurry\\s+forward|quick\\s+forward|rapid\\s+forward|expedite\\s+forward|double\\s+speed|triple\\s+speed|increase\\s+speed)\\b",
            
            "rewind": "\\b(?:rewind|rewinding|rewound|fast\\s+backward|reverse\\s+(?:quickly|fast)|back\\s+up|go\\s+back|reverse\\s+playback|backward\\s+(?:quickly|fast)|retreat|retreating|retreated|regress|regressing|regressed|slow\\s+reverse|reverse\\s+slowly)\\b",
            
            "seek": "\\b(?:seek|seeking|sought|jump\\s+to|go\\s+to|move\\s+to|navigate\\s+to|position\\s+to|scrub\\s+to|advance\\s+to|retreat\\s+to|set\\s+position|change\\s+position|adjust\\s+position|locate\\s+to|find\\s+position|progress\\s+to)\\b",
            
            "mute": "\\b(?:mute|muting|muted|silence|silencing|silenced|quiet|quieting|quieted|turn\\s+off\\s+sound|disable\\s+sound|kill\\s+sound|cut\\s+sound|no\\s+sound|sound\\s+off|audio\\s+off|volume\\s+off|silent\\s+mode)\\b",
            
            "unmute": "\\b(?:unmute|unmuting|unmuted|unsilence|unsilencing|unsilenced|unquiet|unquieting|unquieted|turn\\s+on\\s+sound|enable\\s+sound|restore\\s+sound|bring\\s+back\\s+sound|sound\\s+on|audio\\s+on|volume\\s+on|exit\\s+silent\\s+mode)\\b",
            
            "fullscreen": "\\b(?:full\\s+screen|fullscreen|full-screen|maximize|maximizing|maximized|expand|expanding|expanded|enlarge|enlarging|enlarged|stretch|stretching|stretched|fill\\s+screen|screen\\s+fill|wide\\s+screen|cinema\\s+mode|theater\\s+mode|immersive\\s+mode)\\b",
            
            "captions": "\\b(?:caption|captions|subtitle|subtitles|closed\\s+caption|closed\\s+captions|cc|cc'?s|text|texts|transcript|transcripts|sub|subs|overlay|overlays|dialogue|dialogues|speech\\s+text|spoken\\s+text|audio\\s+description)\\b",
            
            "speed": "\\b(?:speed|speeding|sped|playback\\s+speed|play\\s+speed|rate|rates|pace|pacing|paced|tempo|tempos|tempoing|tempoed|rhythm|rhythms|rhythmical|velocity|velocities|quickness|quicknesses|rapidity|rapidities)\\b",
            
            "shuffle": "\\b(?:shuffle|shuffling|shuffled|random|randomize|randomizing|randomized|mix|mixing|mixed|scramble|scrambling|scrambled|jumble|jumbling|jumbled|disorder|disordering|disordered|rearrange|rearranging|rearranged)\\b",
            
            "repeat": "\\b(?:repeat|repeating|repeated|loop|looping|looped|cycle|cycling|cycled|replay|replaying|replayed|encore|encoring|encored|again|repeat\\s+mode|loop\\s+mode|continuous\\s+play|infinite\\s+play)\\b",
            
            "increase": "\\b(?:increase|increased|increasing|up|higher|raise|raised|raising|boost|boosted|boosting|amplify|amplified|amplifying|enhance|enhanced|enhancing|elevate|elevated|elevating|pump\\s+up|turn\\s+up|crank\\s+up|ramp\\s+up|scale\\s+up|step\\s+up|jack\\s+up|bump\\s+up|push\\s+up|bring\\s+up|make\\s+it\\s+higher|louder|brighter|stronger|more|maximize|max\\s+out|intensify)\\b",
            
            "decrease": "\\b(?:decrease|decreased|decreasing|down|lower|lowered|lowering|reduce|reduced|reducing|diminish|diminished|diminishing|lessen|lessened|lessening|drop|dropped|dropping|cut|cutting|turn\\s+down|bring\\s+down|scale\\s+down|step\\s+down|tone\\s+down|dial\\s+down|wind\\s+down|ramp\\s+down|make\\s+it\\s+lower|quieter|dimmer|weaker|less|minimize|min\\s+out|soften)\\b"
        ]
        
        // Sort actions by pattern specificity (longer patterns first for better matching)
        let sortedMediaActions = mediaActions.sorted { $0.value.count > $1.value.count }
        
        for (action, pattern) in sortedMediaActions {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return action
            }
        }
        
        return nil
    }
    
    private func extractAppAction(text: String) -> String? {
        let appActions: [String: String] = [
            "open": "\\b(?:open|opened|opening|launch|launched|launching|start|show|display|view|access|load|bring\\s+up|pull\\s+up|fire\\s+up|boot|go\\s+to|navigate\\s+to|switch\\s+to|take\\s+me\\s+to|turn\\s+on|on|enable|enabled|activate|activated|power\\s+on|switch\\s+on)\\b",
            "increase": "\\b(?:increase|increased|increasing|up|higher|raise|raised|raising|boost|boosted|boosting|amplify|amplified|amplifying|enhance|enhanced|enhancing|elevate|elevated|elevating|pump\\s+up|turn\\s+up|crank\\s+up|ramp\\s+up|scale\\s+up|step\\s+up|jack\\s+up|bump\\s+up|push\\s+up|bring\\s+up|make\\s+it\\s+higher|louder|brighter|stronger|more|maximize|max\\s+out|intensify)\\b",
            "decrease": "\\b(?:decrease|decreased|decreasing|down|lower|lowered|lowering|reduce|reduced|reducing|diminish|diminished|diminishing|lessen|lessened|lessening|drop|dropped|dropping|cut|cutting|turn\\s+down|bring\\s+down|scale\\s+down|step\\s+down|tone\\s+down|dial\\s+down|wind\\s+down|ramp\\s+down|make\\s+it\\s+lower|quieter|dimmer|weaker|less|minimize|min\\s+out|soften)\\b"
        ]
        // Sort actions by pattern specificity (longer patterns first for better matching)
        let sortedActions = appActions.sorted { $0.value.count > $1.value.count }
        for (action, pattern) in sortedActions {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return action
            }
        }
        return nil
    }
    
    private func extractPhoneAction(text: String) -> String? {
        let phoneActions: [String: String] = [
            "call": "\\b(?:call|calling|called|phone|phoning|phoned|dial|dialing|dialed|ring|ringing|rang)\\b",
            "message": "\\b(?:text|texting|texted|message|messaging|messaged|sms|send\\s+sms|sending\\s+sms|sent\\s+sms)\\b"
        ]
        
        // Sort actions by pattern specificity (longer patterns first for better matching)
        let sortedActions = phoneActions.sorted { $0.value.count > $1.value.count }
        
        for (action, pattern) in sortedActions {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return action
            }
        }
        
        return nil
    }
    
    private func extractTool(text: String) -> String? {
        if text.range(of: "\\b(?:alarm|alarms|wake\\s+up|wake\\s+me|wake\\s+me\\s+up|morning\\s+alarm|set\\s+alarm|alarm\\s+clock|wakeup|wake-up|rouse|rise|get\\s+up|ring|ringer|buzzer|morning\\s+call|wake\\s+call|alert\\s+me|morning\\s+alert|sleep\\s+alarm|snooze|beep)\\b", options: .regularExpression) != nil {
            return "alarm"
        }
        
        if text.range(of: "\\b(?:timer|timers|countdown|count\\s+down|set\\s+timer|start\\s+timer|kitchen\\s+timer|cooking\\s+timer|egg\\s+timer|time\\s+me|timing|timed|set\\s+a\\s+timer|countdown\\s+timer|interval\\s+timer|remind\\s+me\\s+in|alert\\s+in|notify\\s+in|time\\s+limit|duration|time\\s+out)\\b", options: .regularExpression) != nil {
            return "timer"
        }
        
        if text.range(of: "\\b(?:stopwatch|stop\\s+watch|chronometer|lap\\s+timer|lap\\s+time|split\\s+time|time\\s+lap|elapsed\\s+time|running\\s+time|measure\\s+time|track\\s+time|timing|chrono|lap|laps|split|splits|time\\s+it|how\\s+long|duration\\s+tracker)\\b", options: .regularExpression) != nil {
            return "stopwatch"
        }
        
        return nil
    }
    
    
    private func extractActivityType(text: String) -> String? {
        let activities: [String: String] = [
            "outdoor run": "\\b(?:outdoor\\s+)?(?:run|running|ran|jog|jogging|jogged|sprint|sprinting|sprinted|dash|dashing|race|racing|trail\\s+run|trail\\s+running|distance\\s+run|long\\s+run|short\\s+run|tempo\\s+run|interval\\s+run|fartlek|road\\s+run|cross\\s+country|marathon|half\\s+marathon|5k|10k|runner|runners)\\b",
            
            "indoor cycling": "\\b(?:indoor\\s+)?(?:cycling|cycle|cycled|bike|biking|biked|bicycle|bicycling|spin|spinning|spin\\s+class|stationary\\s+bike|exercise\\s+bike|bike\\s+ride|pedal|pedaling|pedalled|indoor\\s+bike|cycle\\s+class|RPM|cadence\\s+training|peloton|zwift|virtual\\s+cycling|turbo\\s+trainer|trainer\\s+ride)\\b",
            
            "swimming": "\\b(?:swim|swimming|swam|swum|swimmer|pool|lap|laps|freestyle|backstroke|breaststroke|butterfly|stroke|strokes|aquatic|water\\s+exercise|lap\\s+swimming|pool\\s+workout|open\\s+water|triathlon\\s+swim|swim\\s+training|water\\s+aerobics|aqua|diving|float|floating)\\b",
            
            "yoga": "\\b(?:yoga|yogi|asana|asanas|meditation|meditate|meditating|meditated|stretch|stretching|stretched|flexibility|vinyasa|hatha|ashtanga|bikram|hot\\s+yoga|power\\s+yoga|yin\\s+yoga|restorative\\s+yoga|pranayama|breathing\\s+exercise|mindfulness|zen|namaste|downward\\s+dog|warrior\\s+pose|sun\\s+salutation|flow|yoga\\s+flow)\\b",
            
            "walking": "\\b(?:walk|walking|walked|walker|stroll|strolling|strolled|hike|hiking|hiked|hiker|trek|trekking|trekked|ramble|rambling|wander|wandering|wandered|amble|ambling|march|marching|power\\s+walk|brisk\\s+walk|leisurely\\s+walk|nature\\s+walk|trail\\s+walk|hill\\s+walk|speed\\s+walk|fitness\\s+walk|evening\\s+walk|morning\\s+walk)\\b",
            
            "workout": "\\b(?:workout|work\\s+out|worked\\s+out|exercise|exercising|exercised|training|train|trained|gym|gymnasium|fitness|fit|strength\\s+training|weight\\s+training|weightlifting|lift|lifting|cardio|HIIT|circuit\\s+training|crossfit|bootcamp|boot\\s+camp|calisthenics|bodyweight|resistance\\s+training|functional\\s+training|core\\s+workout|abs\\s+workout|upper\\s+body|lower\\s+body|full\\s+body)\\b"
        ]
        
        // Sort by pattern length for more specific matching
        let sortedActivities = activities.sorted { $0.value.count > $1.value.count }
        
        for (activity, pattern) in sortedActivities {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return activity
            }
        }
        
        return nil
    }
    
    private func extractApp(text: String) -> String? {
        let apps: [String: String] = [
            "heart rate": "\\b(?:heart\\s+rate|heartrate|heart\\s+beat|heartbeat|pulse|pulse\\s+rate|bpm|beats\\s+per\\s+minute|cardiac|cardiac\\s+rate|heart\\s+rhythm|resting\\s+heart\\s+rate|rhr|max\\s+heart\\s+rate|maximum\\s+heart\\s+rate|heart\\s+health|cardiovascular|cardio|ticker|heart\\s+monitor|heart\\s+sensor|hr|beat|beats|beating|palpitation|palpitations|tachycardia|bradycardia|heart\\s+zone|target\\s+heart\\s+rate|recovery\\s+heart\\s+rate)\\b",
            "blood oxygen": "\\b(?:blood\\s+oxygen|oxygen|o2|spo2|sp\\s+o2|oxygen\\s+saturation|oxygen\\s+level|oxygen\\s+levels|blood\\s+o2|oxygen\\s+sat|o2\\s+sat|o2\\s+level|o2\\s+saturation|pulse\\s+ox|pulse\\s+oximetry|oximeter|oxygen\\s+reading|oxygen\\s+sensor|saturation|sat|blood\\s+oxygen\\s+level|arterial\\s+oxygen|respiratory|respiration|breathing|breath|lung\\s+function|oxygenation|hypoxia|oxygen\\s+content|sp2|SP2)\\b",
            "stress": "\\b(?:stress|stressed|stressful|stress\\s+level|stress\\s+score|stress\\s+index|anxiety|anxious|worried|worry|worrying|tension|tense|pressure|pressured|strain|strained|overwhelm|overwhelmed|nervous|nervousness|burnout|burnt\\s+out|mental\\s+stress|emotional\\s+stress|psychological\\s+stress|chronic\\s+stress|acute\\s+stress|relaxation|relax|calm|calmness|peace|peaceful|tranquil|serene|zen|mindfulness)\\b",
            "brightness": "\\b(?:brightness|bright|brighter|brighten|brightening|screen\\s+brightness|display\\s+brightness|luminosity|luminance|backlight|screen\\s+light|light\\s+level|dim|dimmer|dimming|dimness|darken|darker|darkening|auto\\s+brightness|adaptive\\s+brightness|brightness\\s+level|screen\\s+intensity|display\\s+intensity|illumination|illuminate|glow|glowing|radiance|light\\s+output|ambient\\s+light|screen\\s+glow|visibility|contrast|gamma|exposure|luminous)\\b"
        ]
        // Sort by pattern length for more specific matching
        let sortedApps = apps.sorted { $0.value.count > $1.value.count }
        for (app, pattern) in sortedApps {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return app
            }
        }
        return nil
    }
    
    
    private func extractContact(text: String) -> String? {
        // First check for hardcoded contacts
        let contacts: [String: String] = [
            "mom": "\\b(?:mom|mother|mama|mum|mummy|ma|mommy|mamma|momma)\\b",
            "dad": "\\b(?:dad|father|papa|pop|daddy|pappa|pops|pa|old\\s+man)\\b",
            "sister": "\\b(?:sister|sis|sibling)\\b",
            "brother": "\\b(?:brother|bro|sibling)\\b"
        ]
        
        for (contact, pattern) in contacts {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return contact
            }
        }
        
        // Special case: check for emergency services first
        let emergencyServices: [String: String] = [
            "911": "\\b(?:911|emergency|police|fire\\s+department|ambulance|paramedics)\\b",
            "emergency": "\\b(?:emergency|urgent|help|sos|crisis|disaster|accident|medical\\s+emergency)\\b"
        ]
        
        for (service, pattern) in emergencyServices {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return service
            }
        }
        
        // If no hardcoded contact found, try to extract any name after phone action keywords
        let phoneActionPatterns = [
            "\\b(?:call|calling|called|phone|phoning|phoned|dial|dialing|dialed|ring|ringing|rang|contact|contacting|contacted|reach|reaching|reached|get\\s+in\\s+touch\\s+with|getting\\s+in\\s+touch\\s+with|got\\s+in\\s+touch\\s+with)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)",
            "\\b(?:text|texting|texted|message|messaging|messaged|sms|send\\s+sms\\s+to|sending\\s+sms\\s+to|sent\\s+sms\\s+to|send\\s+message\\s+to|sending\\s+message\\s+to|sent\\s+message\\s+to|write\\s+to|writing\\s+to|wrote\\s+to)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)",
            "\\b(?:video\\s+call|video\\s+calling|video\\s+called|facetime|face\\s+time|skype|skyping|skyped|zoom|zooming|zoomed)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)",
            "([A-Za-z][A-Za-z\\s]{1,30})\\s+(?:please|pls)",
            "(?:hey|hi|hello|yo)\\s+([A-Za-z][A-Za-z\\s]{1,30})",
            "\\b(?:to|for)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)",
            "([A-Za-z][A-Za-z\\s]{1,30})(?:'s|'s)\\s+(?:number|phone|mobile|cell)",
            "\\b(?:contact|person|friend|family|colleague|neighbor|relative)\\s+(?:named|called)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)",
            "([A-Za-z][A-Za-z\\s]{1,30})\\s+(?:from|at)\\s+(?:work|office|home|school|gym|church|club)",
            "\\b(?:dr|doctor|mr|mister|mrs|miss|ms)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)",
            "([A-Za-z][A-Za-z\\s]{1,30})\\s+(?:the|my)\\s+(?:doctor|dentist|lawyer|teacher|boss|manager|supervisor|assistant|secretary|colleague|friend|neighbor|relative|family|cousin|aunt|uncle|grandmother|grandfather|grandma|grandpa|nephew|niece)",
            "\\b(?:my|the)\\s+(?:doctor|dentist|lawyer|teacher|boss|manager|supervisor|assistant|secretary|colleague|friend|neighbor|relative|family|cousin|aunt|uncle|grandmother|grandfather|grandma|grandpa|nephew|niece)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)"
        ]
        
        for pattern in phoneActionPatterns {
            if let range = text.range(of: pattern, options: .regularExpression) {
                let matchedText = String(text[range])
                let matches = try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive]).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                
                if let match = matches.first, match.numberOfRanges > 1 {
                    let nameRange = Range(match.range(at: 1), in: matchedText)!
                    let name = String(matchedText[nameRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                    
                    // Validate extracted name
                    if !name.isEmpty && 
                       name.count >= 2 && 
                       name.count <= 30 &&
                       !["the", "my", "a", "an", "to", "for", "at", "in", "on", "with", "by", "from", "of", "and", "or", "but", "please", "pls", "hey", "hi", "hello", "yo", "now", "today", "tomorrow", "yesterday", "later", "soon", "quickly", "slowly", "again", "back", "home", "work", "office", "number", "phone", "mobile", "cell", "contact", "person", "friend", "family", "someone", "anybody", "anyone", "somebody", "nobody", "everyone", "everybody"].contains(name.lowercased()) {
                        return name
                    }
                }
            }
        }
        
        // Additional pattern specifically for phone numbers with common formats
        let phoneNumberPatterns = [
            "\\b(\\d{3}[-.]?\\d{3}[-.]?\\d{4})\\b",
            "\\b(\\(\\d{3}\\)\\s?\\d{3}[-.]?\\d{4})\\b",
            "\\b(\\+\\d{1,3}\\s?\\d{3,4}\\s?\\d{3,4}\\s?\\d{3,4})\\b",
            "\\b(\\d{10,15})\\b",
            "\\b(?:number|phone|mobile|cell)\\s+(\\d{3}[-.]?\\d{3}[-.]?\\d{4})\\b",
            "\\b(?:number|phone|mobile|cell)\\s+(\\(\\d{3}\\)\\s?\\d{3}[-.]?\\d{4})\\b",
            "\\b(?:call|dial|phone)\\s+(\\d{3}[-.]?\\d{3}[-.]?\\d{4})\\b",
            "\\b(?:call|dial|phone)\\s+(\\(\\d{3}\\)\\s?\\d{3}[-.]?\\d{4})\\b"
        ]
        
        for pattern in phoneNumberPatterns {
            if let range = text.range(of: pattern, options: .regularExpression) {
                let matchedText = String(text[range])
                let matches = try! NSRegularExpression(pattern: "(\\d{3}[-.]?\\d{3}[-.]?\\d{4}|\\(\\d{3}\\)\\s?\\d{3}[-.]?\\d{4}|\\+\\d{1,3}\\s?\\d{3,4}\\s?\\d{3,4}\\s?\\d{3,4}|\\d{10,15})", options: []).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                
                if let match = matches.first {
                    let numberRange = Range(match.range, in: matchedText)!
                    let phoneNumber = String(matchedText[numberRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                    
                    if !phoneNumber.isEmpty {
                        return phoneNumber
                    }
                }
            }
        }
        
        return nil
    }
    
    
    private func extractLocation(text: String) -> String? {
        // First check for hardcoded locations
        let locations: [String: String] = [
            "london": "\\b(?:london|uk|england|britain|great\\s+britain|united\\s+kingdom|british)\\b",
            "bangalore": "\\b(?:bangalore|bengaluru|blr|karnataka|india|indian)\\b",
            "mumbai": "\\b(?:mumbai|bombay|maharashtra|india|indian)\\b",
            "delhi": "\\b(?:delhi|new\\s+delhi|ncr|national\\s+capital\\s+region|india|indian)\\b"
        ]
        
        for (location, pattern) in locations {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return location
            }
        }
        
        // Try to extract location after weather-related keywords
        let weatherPattern = "(?:weather|forecast|temperature|climate|conditions)\\s+(.+?)(?:\\s|$|\\.|\\?|!)"
        if let range = text.range(of: weatherPattern, options: .regularExpression) {
            let matchedText = String(text[range])
            let matches = try! NSRegularExpression(pattern: weatherPattern, options: [.caseInsensitive]).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
            
            if let match = matches.first, match.numberOfRanges > 1 {
                let locationRange = Range(match.range(at: 1), in: matchedText)!
                let extractedLocation = String(matchedText[locationRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                
                if !extractedLocation.isEmpty && 
                   extractedLocation.count >= 2 && 
                   extractedLocation.count <= 50 &&
                   !["in", "at", "for", "the", "my", "current", "here", "there", "today", "tomorrow", "now"].contains(extractedLocation.lowercased()) {
                    return extractedLocation
                }
            }
        }
        
        return "current location"  // Default
    }
    
    private func extractAttribute(text: String) -> String? {
        let attributes: [String: String] = [
            "forecast": "\\b(?:forecast|forecasts|forecasting|prediction|predictions|predict|predicted|outlook|outlooks|projection|projections|future\\s+weather|upcoming\\s+weather|expected\\s+weather|weather\\s+ahead|what'?s\\s+coming|coming\\s+weather|next\\s+days|next\\s+week|tomorrow|later|ahead|anticipate|anticipated|expect|expected|probable|likely|chance\\s+of)\\b",
        
            "temperature": "\\b(?:temperature|temperatures|temp|temps|hot|cold|warm|cool|heat|heated|heating|chill|chilly|freezing|frozen|frost|frosty|degrees|degree|celsius|fahrenheit|thermometer|thermal|feels\\s+like|wind\\s+chill|heat\\s+index|mild|moderate|extreme|scorching|boiling|icy|frigid|lukewarm|toasty)\\b",
            
            "rain": "\\b(?:rain|raining|rainy|rained|rainfall|rainwater|shower|showers|showery|drizzle|drizzling|drizzly|sprinkle|sprinkling|downpour|pouring|pour|umbrella|wet|wetness|damp|dampness|moisture|precipitation|precipitating|storm|stormy|thunderstorm|cloudburst|deluge|mist|misty|monsoon)\\b",
            
            "humidity": "\\b(?:humidity|humid|moisture|moistness|damp|dampness|dank|muggy|sticky|clammy|steamy|sultry|wet|wetness|dew|dew\\s+point|relative\\s+humidity|water\\s+vapor|vapour|atmospheric\\s+moisture|air\\s+moisture|condensation|saturated|saturation|dry|dryness|arid)\\b",
            
            "air quality": "\\b(?:air\\s+quality|aqi|air\\s+quality\\s+index|pollution|polluted|pollutants|smog|smoggy|haze|hazy|particulate|particles|pm2\\.?5|pm10|pm\\s+2\\.?5|pm\\s+10|ozone|allergens|pollen|dust|emissions|exhaust|fumes|toxic|toxins|clean\\s+air|dirty\\s+air|unhealthy\\s+air|breathable|air\\s+pollution)\\b"
        ]
        
        for (attr, pattern) in attributes {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return attr
            }
        }
        
        return nil
    }
    
    private func extractType(text: String) -> String? {
        if text.range(of: "\\b(?:above|over|high|higher|highest|exceed|exceeding|exceeded|exceeds|higher|more\\s+than|greater\\s+than|beyond|surpass|surpassing|surpassed|surpasses|top|topping|topped|tops|beat|beating|beaten|beats|cross|crossing|crossed|crosses|pass|passing|passed|passes)\\b", options: [.regularExpression, .caseInsensitive]) != nil {
            return "high"
        }
        if text.range(of: "\\b(?:below|under|low|lower|lowest|less\\s+than|lower\\s+than|beneath|underneath|drop|drops|dropped|dropping|fall|falls|fell|falling|decrease|decreases|decreased|decreasing|reduce|reduces|reduced|reducing|decline|declines|declined|declining|go\\s+down|goes\\s+down|went\\s+down|going\\s+down|dip|dips|dipped|dipping|plunge|plunges|plunged|plunging|sink|sinks|sank|sinking|tumble|tumbles|tumbled|tumbling)\\b", options: [.regularExpression, .caseInsensitive]) != nil {
            return "low"
        }
        return null
    }
    
    private func extractPeriod(text: String) -> String? {
        let periods: [String: String] = [
            "daily": "\\b(?:daily|every\\s+day|each\\s+day|day\\s+by\\s+day|per\\s+day|on\\s+a\\s+daily\\s+basis|day\\s+to\\s+day|everyday|each\\s+and\\s+every\\s+day)\\b",
            "weekly": "\\b(?:weekly|every\\s+week|each\\s+week|week\\s+by\\s+week|per\\s+week|on\\s+a\\s+weekly\\s+basis|week\\s+to\\s+week|every\\s+seven\\s+days|once\\s+a\\s+week|once\\s+per\\s+week)\\b",
            "monthly": "\\b(?:monthly|every\\s+month|each\\s+month|month\\s+by\\s+month|per\\s+month|on\\s+a\\s+monthly\\s+basis|month\\s+to\\s+month|once\\s+a\\s+month|once\\s+per\\s+month|every\\s+thirty\\s+days)\\b",
            "yearly": "\\b(?:yearly|annually|every\\s+year|each\\s+year|year\\s+by\\s+year|per\\s+year|on\\s+a\\s+yearly\\s+basis|year\\s+to\\s+year|once\\s+a\\s+year|once\\s+per\\s+year|every\\s+365\\s+days)\\b",
            "hourly": "\\b(?:hourly|every\\s+hour|each\\s+hour|hour\\s+by\\s+hour|per\\s+hour|on\\s+an\\s+hourly\\s+basis|hour\\s+to\\s+hour|once\\s+an\\s+hour|once\\s+per\\s+hour|every\\s+sixty\\s+minutes)\\b"
        ]
        
        // Sort by pattern length for more specific matching
        let sortedPeriods = periods.sorted { $0.value.count > $1.value.count }
        
        for (period, pattern) in sortedPeriods {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return period
            }
        }
        
        return nil
    }
    
    private func extractEventType(text: String) -> String? {
        if text.range(of: "\\b(?:weight|weigh|weighing|weighed|kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|pounds?|lbs?|lb|body\\s+weight|body\\s+mass|mass|scale|scales|weighing\\s+scale|weight\\s+scale|heavy|heaviness|light|lightness|bmi|body\\s+mass\\s+index|stone|st|grams?|grammes?|g|ounces?|oz|#)\\b", options: [.regularExpression, .caseInsensitive]) != nil {
            return "weight"
        }
        
        if text.range(of: "\\b(?:menstrual|menstruation|menstruating|menstruate|period|periods|cycle|cycles|monthly\\s+cycle|time\\s+of\\s+month|that\\s+time|aunt\\s+flo|flow|bleeding|spotting|pms|premenstrual|ovulation|ovulating|ovulate|fertile|fertility|fertility\\s+window|luteal\\s+phase|follicular\\s+phase|cramping|cramps|menses|feminine\\s+hygiene|menstrual\\s+health|reproductive\\s+cycle)\\b", options: [.regularExpression, .caseInsensitive]) != nil {
            return "menstrual cycle"
        }
        
        return nil
    }
    
    
    // MARK: - Helper Methods
    
    private func addContextualSlots(text: String, intent: String, slots: inout [String: Any]) {
        switch intent {
        case "QueryPoint":
            // Try to infer missing metric
            if slots["metric"] == nil {
                if let inferredMetric = inferMetricFromContext(text: text) {
                    slots["metric"] = inferredMetric
                }
            }
            
            // Add default time_ref if not present
            if slots["time_ref"] == nil {
                slots["time_ref"] = "today"
            }
            
            // Add default unit based on metric
            if slots["unit"] == nil, let metric = slots["metric"] as? String {
                switch metric {
                case "steps":
                    slots["unit"] = "count"
                case "distance":
                    slots["unit"] = "km"
                case "calories":
                    slots["unit"] = "kcal"
                case "heart rate":
                    slots["unit"] = "bpm"
                case "sleep":
                    slots["unit"] = "hours"
                case "weight":
                    slots["unit"] = "kg"
                case "spo2":
                    slots["unit"] = "percent"
                case "stress":
                    slots["unit"] = "score"
                default:
                    break
                }
            }
            
            // Add default identifier if not present
            if slots["identifier"] == nil {
                slots["identifier"] = "average"
            }
            
        case "SetGoal", "SetThreshold":
            // Add default unit based on metric
            if slots["unit"] == nil, let metric = slots["metric"] as? String {
                switch metric {
                case "steps":
                    slots["unit"] = "count"
                case "distance":
                    slots["unit"] = "km"
                case "calories":
                    slots["unit"] = "kcal"
                case "heart rate":
                    slots["unit"] = "bpm"
                case "sleep":
                    slots["unit"] = "hours"
                case "weight":
                    slots["unit"] = "kg"
                case "spo2":
                    slots["unit"] = "percent"
                case "stress":
                    slots["unit"] = "score"
                default:
                    break
                }
            }
            
        case "TimerStopwatch":
            // For timer/stopwatch, try to extract more specific actions
            if slots["action"] == nil {
                if let timerAction = extractTimerAction(text: text) {
                    slots["action"] = timerAction
                }
            }
            
            // Extract value for timer/alarm setting if not already present
            if slots["value"] == nil {
                if let value = extractValue(text: text, intent: intent) {
                    slots["value"] = value
                }
            }
            
        case "PhoneAction":
            // For phone actions, try to extract more specific actions
            if slots["action"] == nil {
                if let phoneAction = extractPhoneAction(text: text) {
                    slots["action"] = phoneAction
                }
            }
            
        case "MediaAction":
            // For media actions, try to extract more specific actions
            if slots["action"] == nil {
                if let mediaAction = extractMediaAction(text: text) {
                    slots["action"] = mediaAction
                }
            }
            
            // Add default target if action is play
            if slots["target"] == nil, let action = slots["action"] as? String {
                if action == "play" {
                    slots["target"] = "music"
                }
            }
            
        case "OpenApp":
            // For app actions, try to extract more specific actions
            if slots["action"] == nil {
                if let appAction = extractAppAction(text: text) {
                    slots["action"] = appAction
                } else {
                    slots["action"] = "open"  // Default action
                }
            }
            
            // Add default target
            if slots["target"] == nil {
                slots["target"] = "app"
            }
            
        case "LogEvent":
            // Add default value and unit for weight logging
            if slots["value"] == nil && slots["event_type"] as? String == "weight" {
                // Try to extract weight value again with more patterns
                if let value = extractValue(text: text, intent: intent) {
                    slots["value"] = value
                }
            }
            
            // Add default unit based on event type
            if slots["unit"] == nil, let eventType = slots["event_type"] as? String {
                switch eventType {
                case "weight":
                    slots["unit"] = "kg"
                case "water intake":
                    slots["unit"] = "ml"
                case "medication":
                    slots["unit"] = "dose"
                case "blood pressure":
                    slots["unit"] = "mmHg"
                case "blood sugar":
                    slots["unit"] = "mg/dL"
                default:
                    break
                }
            }
            
            // Add default time_ref
            if slots["time_ref"] == nil {
                slots["time_ref"] = "now"
            }
            
        case "WeatherQuery":
            // Add default attribute if not present
            if slots["attribute"] == nil {
                slots["attribute"] = "forecast"
            }
            
        case "QueryTrend":
            // Add default period and unit
            if slots["period"] == nil {
                slots["period"] = "weekly"
            }
            
            if slots["unit"] == nil, let metric = slots["metric"] as? String {
                switch metric {
                case "steps":
                    slots["unit"] = "count"
                case "distance":
                    slots["unit"] = "km"
                case "calories":
                    slots["unit"] = "kcal"
                case "heart rate":
                    slots["unit"] = "bpm"
                case "sleep":
                    slots["unit"] = "hours"
                case "weight":
                    slots["unit"] = "kg"
                case "spo2":
                    slots["unit"] = "percent"
                case "stress":
                    slots["unit"] = "score"
                default:
                    break
                }
            }
            
        case "ToggleFeature":
            if slots["feature"] == nil {
                if let feature = extractFeature(text: text) {
                    slots["feature"] = feature
                }
            }
            if slots["state"] == nil {
                if let state = extractState(text: text) {
                    slots["state"] = state
                }
            }
            
        case "StartActivity", "StopActivity":
            if slots["activity_type"] == nil {
                if let activityType = extractActivityType(text: text) {
                    slots["activity_type"] = activityType
                }
            }
            if slots["time_ref"] == nil {
                slots["time_ref"] = "today"
            }
            
        default:
            break
        }
    }
    
    private func inferMetricFromContext(text: String) -> String? {
        // Pre-compiled regex patterns for better performance
        let inferencePatterns: [String: [String]] = [
            "steps": [
                "\\b(?:walk|walked|walking)\\b(?!\\s+distance)",
                "\\bhow\\s+much.*(?:walk|walked)\\b",
                "\\bsteps?\\b",
                "\\bpace|paces|stride|strides|footsteps?\\b",
                "\\bmove|moved|movement|activity\\b"
            ],
            "distance": [
                "\\bhow\\s+far\\b",
                "\\b(?:walk|walked|walking)\\s+(?:distance|far)\\b",
                "\\bdistance.*(?:walk|walked)\\b",
                "\\bkilometers?\\b|\\bmiles?\\b|\\bkm\\b",
                "\\btravelled?|traveled|covered|journey|route|path\\b"
            ],
            "heart rate": [
                "\\bheart\\s+rate\\b|\\bheartrate\\b|\\bpulse\\b|\\bhr\\b|\\bbpm\\b",
                "\\bcardiac|cardio|cardiovascular\\b",
                "\\bbeats?\\s+per\\s+minute|rhythm\\b"
            ],
            "calories": [
                "\\bcalories?\\b|\\bkcal\\b|\\benergy\\b|\\bburn\\b",
                "\\bfat\\s+burn|metabolic|metabolism\\b",
                "\\bexpended|consumed|intake\\b"
            ],
            "sleep": [
                "\\bsleep|slept|sleeping|rest|rested\\b",
                "\\bnap|napped|napping|slumber\\b",
                "\\bbedtime|night\\s+sleep|sleep\\s+time\\b"
            ],
            "sleep score": [
                "\\bsleep\\s+(?:quality|score|rating|performance|analysis|efficiency)\\b",
                "\\bhow\\s+well\\s+(?:slept|sleep)\\b"
            ],
            "spo2": [
                "\\boxygen|o2|spo2|saturation|blood\\s+oxygen\\b",
                "\\bpulse\\s+ox|oximeter|breathing\\b"
            ],
            "weight": [
                "\\bweight|weigh|weighing|weighed\\b",
                "\\bbody\\s+mass|bmi|scale\\b",
                "\\bkg|kilogram|pounds?|lbs?\\b"
            ],
            "stress": [
                "\\bstress|stressed|anxiety|anxious\\b",
                "\\btension|worried|overwhelmed\\b",
                "\\bmental\\s+health|relaxation|calm\\b"
            ]
        ]
        
        // Sort by specificity (more specific patterns first)
        let sortedMetrics = inferencePatterns.sorted { $0.value.joined().count > $1.value.joined().count }
        
        for (metric, patterns) in sortedMetrics {
            for pattern in patterns {
                if text.range(of: pattern, options: .regularExpression) != nil {
                    return metric
                }
            }
        }
        
        return nil
    }
}