import { Module } from "@nestjs/common";
import { FoodInfoController } from "./food-info.controller";
import { FoodInfoService } from "./food-info.service";
import { FoodInfoRepository } from "./food-info.repository";

@Module({
  controllers: [FoodInfoController],
  providers: [FoodInfoService, FoodInfoRepository],
})
export class FoodInfoModule {}
