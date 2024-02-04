import { Module, UseFilters } from "@nestjs/common";
import { AuthModule } from "./auth/auth.module";
import { PassportModule } from "@nestjs/passport";
import { CumulativeRecordModule } from "./cumulative-record/cumulative-record.module";
import { typeORMConfig } from "./config/database/typeorm.config";
import { TypeOrmModule } from "@nestjs/typeorm";
import { FoodInfoApiModule } from "./food-info-api/food-info-api.module";
import { ScheduleModule } from "@nestjs/schedule";
import { ConfigModule } from "@nestjs/config";
import { FoodInfoModule } from "./food-info/food-info.module";
import * as dotenv from "dotenv";
import { UserModule } from "./user/user.module";
import { ImageModule } from "./image/image.module";
import { FeedbackModule } from "./feedback/feedback.module";
import { RecordModule } from "./record/record.module";
import { MailerModule } from "@nestjs-modules/mailer";
dotenv.config();
console.log(typeORMConfig);

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    TypeOrmModule.forRoot(typeORMConfig),
    PassportModule.register({ session: true }),
    AuthModule,
    UserModule,
    RecordModule,
    CumulativeRecordModule,
    FoodInfoApiModule,
    ScheduleModule.forRoot(),
    FoodInfoModule,
    FeedbackModule,
    ImageModule,
    MailerModule.forRoot({
      transport: {  // 메일 전송 설정의 정의하는 객체이다.
        host: "smtp.gmail.com", // 메일을 전송할 서버를 Gmail의 SMTP 서버로 설정한다.
        port: 587, // Gmail의 SMTP 서버의 포트는 587이다. 보안 SMTP 서버의 표준포트는 587이다.
        secure: false, //TLS를 사용하지 않음을 의미한다.
        auth: {
          user: process.env.MAIL_USER,
          pass: process.env.MAIL_2STEP_VERIFICATION_KEY,
          // 2step verification을 사용하는 경우에는 앱 비밀번호를 사용해야 한다.
        },
        // 만약 auth를 설정에 실패한다면 아래와 같이 설정한다.
        defaults: {from: '"nest-modules" <modules@nestjs.com>'} // 기본적으로 보내는 사람의 메일 주소를 설정한다.
      },
    }),
  ],
})
export class AppModule {
}
