import { Module } from "@nestjs/common";
import { CumulativeRecordController } from "./cumulative-record.controller";
import { CumulativeRecordService } from "./cumulative-record.service";
import { CumulativeRecordRepository } from "./cumulative.repository";
import { HealthInfoRepository } from "src/user/repositories/health-info.repository";
import { ImageRepository } from "src/image/repositories/image.repository";

@Module({
  controllers: [CumulativeRecordController],
  providers: [
    CumulativeRecordService,
    CumulativeRecordRepository,
    HealthInfoRepository,
    ImageRepository,
  ],
})
export class CumulativeRecordModule {}
