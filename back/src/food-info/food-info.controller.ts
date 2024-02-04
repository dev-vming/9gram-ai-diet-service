import {
  Body,
  Controller,
  Get,
  NotFoundException,
  Post,
  Query,
  UseGuards,
} from "@nestjs/common";
import { FoodInfoService } from "./food-info.service";
import { ApiBody, ApiOperation, ApiTags } from "@nestjs/swagger";
import { isLoggedInGuard } from "src/auth/utils/guards/isLoggedin.guard";
import { FoodNamePipe } from "./pipe/food-name.pipe";

@Controller("food-info")
@ApiTags("Food Info API")
export class FoodInfoController {
  constructor(private foodInfoService: FoodInfoService) {}

  @Get("/foods")
  @UseGuards(isLoggedInGuard)
  @ApiOperation({
    summary: "음식 조회하기",
    description:
      "keyword&lastFoodId: 음식 검색 리스트를 조회한다. foodName: AI로 인식된 음식의 영양성분을 조회한다. foodInfoId: 음식 리스트에서 선택한 음식의 영양성분을 조회한다.",
  })
  async getFoodInfo(
    @Query("keyword", FoodNamePipe) keyword: string,
    @Query("lastFoodId") lastFoodId: string,
    @Query("foodName", FoodNamePipe) foodName: string,
    @Query("foodInfoId") foodInfoId: string
  ) {
    try {
      if (keyword) {
        if (!lastFoodId) {
          const foodListResult =
            await this.foodInfoService.getFoodList(keyword);
          return foodListResult;
        } else {
          const foodNextListResult = await this.foodInfoService.getFoodNextList(
            keyword,
            lastFoodId
          );
          return foodNextListResult;
        }
      }
      // 리스트에 없던 경우
      if (foodName) {
        const foodInfoResult = await this.foodInfoService.getFoodInfo(foodName);
        if (!foodInfoResult) {
          return [];
        }
        return foodInfoResult;
      }
      // 리스트에 있는 경우
      if (foodInfoId) {
        const foodInfoResult =
          await this.foodInfoService.getFoodInfoById(foodInfoId);
        if (!foodInfoResult) {
          return [];
        }
        return foodInfoResult;
      }
    } catch (error) {
      return error;
    }
  }

  @Post("/foods")
  @UseGuards(isLoggedInGuard)
  @ApiOperation({
    summary: "음식들 foodInfoId 리스트 조회하기",
    description: "AI로 인식된 음식들의 foodInfoId를 조회한다.",
  })
  async getFoodList(@Body("foodList", FoodNamePipe) foodList: string[]) {
    try {
      const foodIdListResult =
        await this.foodInfoService.getFoodIdList(foodList);
      return { foodInfoIdList: foodIdListResult };
    } catch (error) {
      return error;
    }
  }
}
