import { TypeOrmModuleOptions } from "@nestjs/typeorm";
import * as config from "config";
import * as dotenv from "dotenv";
const SnakeNamingStrategy =
  require("typeorm-naming-strategies").SnakeNamingStrategy;

const dbConfig = config.get("db");
dotenv.config();

export const typeORMConfig: TypeOrmModuleOptions = {
  type: dbConfig.type,
  host: process.env.DB_HOSTNAME, // || dbConfig.host,
  port: Number(process.env.DB_PORT), // || dbConfig.post,
  username: process.env.DB_USERNAME, // || dbConfig.username,
  password: process.env.DB_PASSWORD, // || dbConfig.password,
  database: process.env.DB_NAME, // || dbConfig.database,
  entities: [__dirname + "/../../**/*.entity.{js, ts}"],
  logging: true,
  synchronize: false,
  namingStrategy: new SnakeNamingStrategy(),
  ssl: {
    rejectUnauthorized: false,
  },
};

