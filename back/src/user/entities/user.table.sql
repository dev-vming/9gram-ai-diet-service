CREATE TABLE "user" (
    "user_id" varchar(36),
    "email" varchar(50) NOT NULL UNIQUE,
    "provider_id" varchar(50) UNIQUE,
    "password" varchar(200),
    "username" varchar(50) NOT NULL UNIQUE,
    "birth_day" date,
    "gender" varchar(50),
    "profile_image" varchar(50),
    "membership" boolean DEFAULT false,
    "created_date" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_date" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "deletedat" timestamp with time zone,
    "health_info_id" varchar(36),
    PRIMARY KEY ("user_id"),
    FOREIGN KEY ("health_info_id") REFERENCES "health_info"("health_info_id")
);
