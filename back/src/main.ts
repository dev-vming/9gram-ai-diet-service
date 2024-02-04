import { NestFactory } from "@nestjs/core";
import { AppModule } from "./app.module";
import { setupSwagger } from "./util/swagger";
import { ValidationPipe } from "@nestjs/common";
import * as dotenv from "dotenv";
import * as session from "express-session";
import * as passport from "passport";
import * as cookieParser from "cookie-parser";
import * as cors from "cors";

dotenv.config({ path: __dirname + "/../.env" });

async function bootstrap() {
  // app 생성
  const app = await NestFactory.create(AppModule);

  // 세션 설정
  app.use(
    session({
      secret: process.env.SESSION_SECRET,
      saveUninitialized: false,
      resave: false,
      cookie: {
        maxAge: 60 * 60 * 1000 * 24, // 24시간
      },
      name: process.env.SESSION_COOKIE_NAME,
    })
  );

  // passport 설정
  app.use(passport.initialize());
  app.use(passport.session());

  // api 프리픽스 설정
  app.setGlobalPrefix("/api");

  // swagger 설정
  setupSwagger(app);

  //cookie 설정
  app.use(cookieParser(process.env.SESSION_SECRET));

  // dto 유효성 검사 설정
  app.useGlobalPipes(
    new ValidationPipe({
      // whitelist: true, // dto에 없는 property는 거름, 200 OK
      // transform: true, // string -> number, 200 OK
      forbidNonWhitelisted: true, // dto에 없는 property는 거름, 400 에러
      transformOptions: {
        enableImplicitConversion: true, // string -> number, 암묵적 형변환 200 OK
      },
    })
  );

  // cors 설정
  app.use(
    cors({
      origin: `${process.env.CLIENT_BASE_URL}`,
      credentials: true,
    })
  );

  // app 실행
  await app.listen(5001);
}
bootstrap();
