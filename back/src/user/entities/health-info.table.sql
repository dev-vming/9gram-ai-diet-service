CREATE TABLE "health_info" (
    "health_info_id" varchar(36),
    "weight" int,
    "height" int,
    "target_weight" int,
    "goal" varchar(50),
    "activity_amount" varchar(50),
    "target_calories" int,
    "recommend_intake" int[],
    "updated_date" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "deletedat" timestamp with time zone,
    PRIMARY KEY ("health_info_id")
);
